#include "annotator.h"
#include "detect.h"
#include "document.h"
#include "embedders/ft_embedder.h"
#include "embedders/tfidf_embedder.h"
#include "embedders/torch_embedder.h"
#include "nasty.h"
#include "thread_pool.h"
#include "timer.h"
#include "util.h"

#include <boost/algorithm/string/join.hpp>
#include <tinyxml2/tinyxml2.h>
#include <pqxx/pqxx>
#include <toml++/toml.hpp>
#include <optional>

static std::unique_ptr<TEmbedder> LoadEmbedder(tg::TEmbedderConfig config) {
    if (config.type() == tg::ET_FASTTEXT) {
        return std::make_unique<TFastTextEmbedder>(config);
    } else if (config.type() == tg::ET_TORCH) {
        return std::make_unique<TTorchEmbedder>(config);
    } else if (config.type() == tg::ET_TFIDF) {
        return std::make_unique<TTfIdfEmbedder>(config);
    } else {
        ENSURE(false, "Bad embedder type");
    }
}

TAnnotator::TAnnotator(
    const std::string& configPath,
    const std::vector<std::string>& languages,
    bool saveNotNews /*= false*/,
    const std::string& mode /* = top */
)
    : Tokenizer(onmt::Tokenizer::Mode::Conservative, onmt::Tokenizer::Flags::CaseFeature)
    , SaveNotNews(saveNotNews)
    , Mode(mode)
{
    ::ParseConfig(configPath, Config);
    SaveTexts = Config.save_texts() || (Mode == "json");
    SaveNotNews = Config.save_not_news() || SaveNotNews;
    ComputeNasty = Config.compute_nasty();

    LOG_DEBUG("Loading models...");

    LanguageDetector.loadModel(Config.lang_detect());
    LOG_DEBUG("FastText language detector loaded");

    for (const std::string& l : languages) {
        tg::ELanguage language = FromString<tg::ELanguage>(l);
        Languages.insert(language);
    }

    for (const auto& modelConfig : Config.category_models()) {
        const tg::ELanguage language = modelConfig.language();
        // Do not load models for languages that are not presented in CLI param
        if (Languages.find(language) == Languages.end()) {
            continue;
        }
        CategoryDetectors[language].loadModel(modelConfig.path());
        LOG_DEBUG("FastText " << ToString(language) << " category detector loaded");
    }

    for (const auto& embedderConfig : Config.embedders()) {
        tg::ELanguage language = embedderConfig.language();
        if (Languages.find(language) == Languages.end()) {
            continue;
        }
        tg::EEmbeddingKey embeddingKey = embedderConfig.embedding_key();
        Embedders[{language, embeddingKey}] = LoadEmbedder(embedderConfig);
    }
}

std::vector<TDbDocument> TAnnotator::AnnotateAll(
    const std::vector<std::string>& fileNames,
    tg::EInputFormat inputFormat) const
{
    TThreadPool threadPool;
    std::vector<TDbDocument> docs;
    std::vector<std::future<std::optional<TDbDocument>>> futures;
    if (inputFormat == tg::IF_JSON) {
        std::vector<TDocument> parsedDocs;
        for (const std::string& path: fileNames) {
            std::ifstream fileStream(path);
            nlohmann::json json;
            fileStream >> json;
            for (const nlohmann::json& obj : json) {
                parsedDocs.emplace_back(obj);
            }
        }
        parsedDocs.shrink_to_fit();
        docs.reserve(parsedDocs.size());
        futures.reserve(parsedDocs.size());
        for (const TDocument& parsedDoc: parsedDocs) {
            futures.push_back(threadPool.enqueue(&TAnnotator::AnnotateDocument, this, parsedDoc));
        }
    } else if (inputFormat == tg::IF_JSONL) {
        std::vector<TDocument> parsedDocs;
        for (const std::string& path: fileNames) {
            std::ifstream fileStream(path);
            std::string record;
            while (std::getline(fileStream, record)) {
                parsedDocs.emplace_back(nlohmann::json::parse(record));
            }
        }
        parsedDocs.shrink_to_fit();
        docs.reserve(parsedDocs.size());
        futures.reserve(parsedDocs.size());
        for (const TDocument& parsedDoc: parsedDocs) {
            futures.push_back(threadPool.enqueue(&TAnnotator::AnnotateDocument, this, parsedDoc));
        }
    } else if (inputFormat == tg::IF_HTML) {
        docs.reserve(fileNames.size());
        futures.reserve(fileNames.size());
        for (const std::string& path: fileNames) {
            using TFunc = std::optional<TDbDocument>(TAnnotator::*)(const std::string&) const;
            futures.push_back(threadPool.enqueue<TFunc>(&TAnnotator::AnnotateHtml, this, path));
        }
    } else if (inputFormat == tg::IF_DB) {
        std::vector<TDocument> parsedDocs;

        // Подключение к БД
        try {
            LOG_DEBUG("Config file " << fileNames[0]);
            auto settings = toml::parse_file(fileNames[0]);
            // Устанавливаем соединение с БД
            
            LOG_DEBUG("settings['Database']['host']: " << settings["Database"]["host"]);
            LOG_DEBUG("settings['Database']['port']: " << settings["Database"]["port"]);
            LOG_DEBUG("settings['Database']['dbname']: " << settings["Database"]["dbname"]);
            LOG_DEBUG("settings['Database']['user']: " << settings["Database"]["user"]);
            LOG_DEBUG("settings['Database']['password']: " << settings["Database"]["password"]);
            LOG_DEBUG("settings['Database']['table']: " << settings["Database"]["table"].value_or("NONE"));
            LOG_DEBUG("settings['Database']['postId']: " << settings["Database"]["postId"]);
            LOG_DEBUG("settings['Database']['startDate']: " << settings["Database"]["startDate"]);

            pqxx::connection conn(
                "dbname=" + std::string(settings["Database"]["dbname"].value_or("")) +
                " user=" + std::string(settings["Database"]["user"].value_or("")) +
                " password=" + std::string(settings["Database"]["password"].value_or("")) +
                " host=" + std::string(settings["Database"]["host"].value_or("")) +
                " port=" + std::string(settings["Database"]["port"].value_or(""))
            );

            pqxx::work txn(conn);
            std::string query;
            txn.exec("SET lc_numeric TO 'C';");
            if(settings["Database"]["table"].value_or("")==""){
                // SQL-запрос
                std::ostringstream os_query;
                os_query << "SELECT "
                      << "post_data.post_id, "
                      << "post_data.post_date, "
                      << "post_data.post_text, "
                      << "post_data.source_id, "
                      << "post_data.source_name, "
                      << "post_data.source_links, "
                      << "post_data.source_type "
                      << "FROM ("
                      << "SELECT DISTINCT ON (tg.\"TelegramId\") "
                      << "    tg.\"Id\" AS post_id, "
                      << "    EXTRACT(EPOCH FROM tg.\"Date\") AS post_date, "
                      << "    tg.\"Text\" AS post_text, "
                      << "    tg.\"SourceId\" AS source_id, "
                      << "    s.\"Name\" AS source_name, "
                      << "    s.\"LinkUrl\" AS source_links, "
                      << "    s.\"SourceType\" AS source_type "
                      << "FROM \"TelegramPost\" tg "
                      << "LEFT JOIN \"Source\" s ON tg.\"SourceId\" = s.\"Id\" "
                      << "WHERE tg.\"Date\" >= '" << settings["Database"]["startDate"].value_or("") << "' "
                      << "UNION ALL "
                      << "SELECT DISTINCT ON (ok.\"OkId\") "
                      << "    ok.\"Id\" AS post_id, "
                      << "    EXTRACT(EPOCH FROM ok.\"Date\") AS post_date, "
                      << "    ok.\"Text\" AS post_text, "
                      << "    ok.\"SourceId\" AS source_id, "
                      << "    s.\"Name\" AS source_name, "
                      << "    s.\"LinkUrl\" AS source_links, "
                      << "    s.\"SourceType\" AS source_type "
                      << "FROM \"OkPost\" ok "
                      << "LEFT JOIN \"Source\" s ON ok.\"SourceId\" = s.\"Id\" "
                      << "WHERE ok.\"Date\" >= '" << settings["Database"]["startDate"].value_or("") << "' "
                      << "UNION ALL "
                      << "SELECT DISTINCT ON (vk.\"VkId\") "
                      << "    vk.\"Id\" AS post_id, "
                      << "    EXTRACT(EPOCH FROM vk.\"Date\") AS post_date, "
                      << "    vk.\"Text\" AS post_text, "
                      << "    vk.\"SourceId\" AS source_id, "
                      << "    s.\"Name\" AS source_name, "
                      << "    s.\"LinkUrl\" AS source_links, "
                      << "    s.\"SourceType\" AS source_type "
                      << "FROM \"VkPost\" vk "
                      << "LEFT JOIN \"Source\" s ON vk.\"SourceId\" = s.\"Id\" "
                      << "WHERE vk.\"Date\" >= '" << settings["Database"]["startDate"].value_or("") << "') AS post_data";

                query = os_query.str();

            }else{
                // SQL-запрос
                query = 
                    "SELECT DISTINCT ON(p.\""+std::string(settings["Database"]["postId"].value_or(""))+"\") \
                    p.\"Id\" AS post_id, \
                    EXTRACT(EPOCH FROM p.\"Date\") AS post_date, \
                    p.\"Text\" AS post_text, \
                    p.\"SourceId\" AS source_id, \
                    s.\"Name\" AS source_name, \
                    s.\"LinkUrl\" AS source_links, \
                    s.\"SourceType\" AS source_type\
                    FROM \"" + std::string(settings["Database"]["table"].value_or("")) + "\" p \
                    LEFT JOIN \"Source\" s ON p.\"SourceId\" = s.\"Id\" \
                    WHERE p.\"Date\" >= '" + std::string(settings["Database"]["startDate"].value_or("")) + "'";
            }
            
            LOG_DEBUG("QUERY:\n"+query)
            pqxx::result rows = txn.exec(query);

            // Обработка результатов запроса
            for (const auto& row : rows) {
                std::string source_links = row["source_links"].c_str(); // PostgreSQL массив как строка
                std::string url = "";

                if (!source_links.empty()) {
                    // Убираем фигурные скобки и разделяем элементы
                    source_links = source_links.substr(1, source_links.size() - 2); // Удаляем '{' и '}'
                    std::stringstream ss(source_links);
                    std::string link;
                    if (std::getline(ss, link, ',')) {
                        url = link; // Первая ссылка
                    }
                }
                nlohmann::json jsonItem = {
                    {"category", ""},
                    {"timestamp", row["post_date"].as<double>()},
                    {"description", std::string(row["post_text"].c_str())},
                    {"is_news", true},
                    {"language", ""},
                    {"out_links", nlohmann::json::array()},
                    {"site_name", std::string(row["source_name"].c_str())},
                    {"text", std::string(row["post_text"].c_str())},
                    {"title", std::string(row["post_text"].c_str())},
                    {"file_name", std::string(row["post_id"].c_str()) + "\;" + std::string(row["source_type"].c_str())},
                    {"url", url}
                };

                parsedDocs.emplace_back(jsonItem);
            }

        } catch (const std::exception& e) {
            std::cerr << "Database error: " << e.what() << std::endl;
            return {};
        }

        // Сжимаем вектор
        parsedDocs.shrink_to_fit();

        // Резервируем место в doc и futures
        docs.reserve(parsedDocs.size());
        futures.reserve(parsedDocs.size());

        // Передаем в пул потоков для обработки
        for (const TDocument& parsedDoc : parsedDocs) {
            futures.push_back(threadPool.enqueue(&TAnnotator::AnnotateDocument, this, parsedDoc));
        }
    } else {
        ENSURE(false, "Bad input format");
    }
    for (auto& futureDoc : futures) {
        std::optional<TDbDocument> doc = futureDoc.get();
        if (!doc) {
            continue;
        }
        if (Languages.find(doc->Language) == Languages.end()) {
            continue;
        }
        if (!doc->IsFullyIndexed()) {
            continue;
        }
        if (!doc->IsNews() && !SaveNotNews) {
            continue;
        }

        docs.push_back(std::move(doc.value()));
    }
    futures.clear();
    docs.shrink_to_fit();
    return docs;
}

std::optional<TDbDocument> TAnnotator::AnnotateHtml(const std::string& path) const {
    std::optional<TDocument> parsedDoc = ParseHtml(path);
    return parsedDoc ? AnnotateDocument(*parsedDoc) : std::nullopt;
}

std::optional<TDbDocument> TAnnotator::AnnotateHtml(const tinyxml2::XMLDocument& html, const std::string& fileName) const {
    std::optional<TDocument> parsedDoc = ParseHtml(html, fileName);
    return parsedDoc ? AnnotateDocument(*parsedDoc) : std::nullopt;
}

std::optional<TDbDocument> TAnnotator::AnnotateDocument(const TDocument& document) const {
    TDbDocument dbDoc;
    dbDoc.Language = DetectLanguage(LanguageDetector, document);
    dbDoc.Url = document.Url;
    dbDoc.Host = GetHost(dbDoc.Url);
    dbDoc.SiteName = document.SiteName;
    dbDoc.Title = document.Title;
    dbDoc.FetchTime = document.FetchTime;
    dbDoc.PubTime = document.PubTime;
    dbDoc.FileName = document.FileName;

    if (SaveTexts) {
        dbDoc.Text = document.Text;
        dbDoc.Description = document.Description;
        dbDoc.OutLinks = document.OutLinks;
    }

    if (Mode == "languages") {
        return dbDoc;
    }

    if (document.Text.length() < Config.min_text_length()) {
        return dbDoc;
    }

    std::string cleanTitle = PreprocessText(document.Title);
    std::string cleanText = PreprocessText(document.Text);

    auto detectorIt = CategoryDetectors.find(dbDoc.Language);
    if (detectorIt != CategoryDetectors.end()) {
        const auto& detector = detectorIt->second;
        dbDoc.Category = DetectCategory(detector, cleanTitle, cleanText);
    }
    for (const auto& [pair, embedder]: Embedders) {
        const auto& [language, embeddingKey] = pair;
        if (language != dbDoc.Language) {
            continue;
        }
        TDbDocument::TEmbedding value = embedder->CalcEmbedding(cleanTitle, cleanText);
        dbDoc.Embeddings.emplace(embeddingKey, std::move(value));
    }
    if (ComputeNasty) {
        dbDoc.Nasty = ComputeDocumentNasty(dbDoc);
    }

    return dbDoc;
}

std::optional<TDocument> TAnnotator::ParseHtml(const std::string& path) const {
    TDocument doc;
    try {
        doc.FromHtml(path.c_str(), Config.parse_links());
    } catch (...) {
        LOG_DEBUG("Bad html: " << path);
        return std::nullopt;
    }
    return doc;
}

std::optional<TDocument> TAnnotator::ParseHtml(const tinyxml2::XMLDocument& html, const std::string& fileName) const {
    TDocument doc;
    try {
        doc.FromHtml(html, fileName, Config.parse_links());
    } catch (...) {
        LOG_DEBUG("Bad html: " << fileName);
        return std::nullopt;
    }
    return doc;
}

std::string TAnnotator::PreprocessText(const std::string& text) const {
    std::vector<std::string> tokens;
    Tokenizer.tokenize(text, tokens);
    return boost::join(tokens, " ");
}
