#include "common.h"
#include "llama.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

// Helper function to create a weighted average of embeddings
static std::vector<float> create_weighted_embedding(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& weights) {
    
    if (embeddings.empty() || weights.empty() || embeddings.size() != weights.size()) {
        throw std::runtime_error("Invalid embeddings or weights");
    }
    
    const size_t n_embd = embeddings[0].size();
    std::vector<float> result(n_embd, 0.0f);
    
    // Compute weighted average
    for (size_t i = 0; i < embeddings.size(); i++) {
        for (size_t j = 0; j < n_embd; j++) {
            result[j] += weights[i] * embeddings[i][j];
        }
    }
    
    return result;
}

// Print the tokens and their texts
static void print_tokens(const llama_vocab* vocab, const std::vector<llama_token>& tokens, const char* name) {
    printf("%s tokens (%zu):\n", name, tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        char token_text[32] = {0};
        llama_token_to_piece(vocab, tokens[i], token_text, sizeof(token_text), 0, true);
        printf("  %zu: %6d '%s'\n", i, (int)tokens[i], token_text);
    }
    printf("\n");
}

static void print_usage(int argc, char** argv) {
    printf("\nUsage: %s -m <model_path> [-ngl <n_gpu_layers>] [-c <context_size>]\n\n", argv[0]);
    printf("  -m <model_path>: Path to the model file (required)\n");
    printf("  -ngl <n_gpu_layers>: Number of GPU layers to use (default: 0)\n");
    printf("  -c <context_size>: Context size (default: 2048)\n\n");
}

int main(int argc, char** argv) {
    // Initialize llama.cpp
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    
    // Default parameters
    std::string model_path;
    int n_gpu_layers = 0;
    int n_ctx = 2048;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-ngl") == 0) {
            if (i + 1 < argc) {
                try {
                    n_gpu_layers = std::stoi(argv[++i]);
                } catch (...) {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) {
                try {
                    n_ctx = std::stoi(argv[++i]);
                } catch (...) {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else {
            print_usage(argc, argv);
            return 1;
        }
    }
    
    if (model_path.empty()) {
        fprintf(stderr, "Model path is required\n");
        print_usage(argc, argv);
        return 1;
    }
    
    // Apply params
    model_params.n_gpu_layers = n_gpu_layers;
    ctx_params.n_ctx = n_ctx;
    ctx_params.embeddings = true;  // Enable embeddings mode
    
    printf("Loading model: %s\n", model_path.c_str());
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }
    
    // Create context for inference
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    
    printf("Embedding size: %d\n", n_embd);
    
    // Tokenize Gettysburg Address first part
    std::string gettysburg = "Four score and seven years ago our fathers brought forth";
    std::vector<llama_token> gettysburg_tokens;
    {
        gettysburg_tokens.resize(32); // Reserve space
        int n_tokens = llama_tokenize(vocab, gettysburg.c_str(), gettysburg.length(), 
                                      gettysburg_tokens.data(), gettysburg_tokens.size(), 
                                      false, false);
        if (n_tokens < 0) {
            n_tokens = -n_tokens;
            gettysburg_tokens.resize(n_tokens);
            llama_tokenize(vocab, gettysburg.c_str(), gettysburg.length(), 
                           gettysburg_tokens.data(), gettysburg_tokens.size(), 
                           false, false);
        } else {
            gettysburg_tokens.resize(n_tokens);
        }
    }
    
    // Create number sequence tokens using different formats to ensure unique tokens
    std::vector<llama_token> number_tokens;
    {
        // Try various formats for numbers to find ones that tokenize differently
        const char* formats[] = {
            "Number %d",      // Format as "Number X"
            "Count to %d",    // Format as "Count to X"
            "%d.",            // Format as "X."
            "%d,",            // Format as "X,"
            "(%d)",           // Format as "(X)"
            "=%d=",           // Format as "=X="
        };
        
        const size_t n_formats = sizeof(formats) / sizeof(formats[0]);
        
        // First try to find a format that gives unique tokens
        bool found_unique_format = false;
        size_t format_idx = 0;
        std::vector<llama_token> temp_tokens;
        
        for (format_idx = 0; format_idx < n_formats && !found_unique_format; format_idx++) {
            printf("Trying format: %s\n", formats[format_idx]);
            temp_tokens.clear();
            bool all_unique = true;
            
            for (int i = 1; i <= 10; i++) {
                char formatted_num[64];
                snprintf(formatted_num, sizeof(formatted_num), formats[format_idx], i);
                
                std::vector<llama_token> tokens(8);
                int n_tokens = llama_tokenize(vocab, formatted_num, strlen(formatted_num),
                                              tokens.data(), tokens.size(),
                                              false, false);
                
                if (n_tokens > 0) {
                    tokens.resize(n_tokens);
                    
                    // Check the first non-space token 
                    llama_token token_to_use = tokens[0];
                    
                    // Verify it's unique
                    if (std::find(temp_tokens.begin(), temp_tokens.end(), token_to_use) != temp_tokens.end()) {
                        all_unique = false;
                        printf("  Format leads to duplicate tokens for number %d\n", i);
                        break;
                    }
                    
                    temp_tokens.push_back(token_to_use);
                    
                    char token_text[32] = {0};
                    llama_token_to_piece(vocab, token_to_use, token_text, sizeof(token_text), 0, true);
                    printf("  Number %d -> token %d ('%s')\n", i, (int)token_to_use, token_text);
                } else {
                    all_unique = false;
                    fprintf(stderr, "  Failed to tokenize number %d with format %s\n", i, formats[format_idx]);
                    break;
                }
            }
            
            if (all_unique) {
                found_unique_format = true;
                printf("Found format with unique tokens: %s\n", formats[format_idx]);
                break;
            }
        }
        
        // If no format worked, try alternative tokenization approaches
        if (!found_unique_format) {
            printf("No format with unique tokens found. Trying alternative approaches.\n");
            
            // Try several different formatting options for numbers
            std::vector<std::string> number_formats = {
                " %d ",    // Space before and after
                "  %d  ",  // Double spaces
                "%d ",     // Space after only
                "[%d]",    // Brackets
                "<%d>",    // Angle brackets
                "-%d-",    // Dashes
                "%d:",     // With colon
                "%d;",     // With semicolon
                "%d!",     // With exclamation
                "%d?",     // With question mark
            };
            
            bool success = false;
            
            // Try each format until we find one that works
            for (const auto& format : number_formats) {
                printf("Trying alternative format: '%s'\n", format.c_str());
                
                temp_tokens.clear();
                bool all_unique = true;
                
                for (int i = 1; i <= 10; i++) {
                    char formatted_num[64];
                    snprintf(formatted_num, sizeof(formatted_num), format.c_str(), i);
                    
                    std::vector<llama_token> tokens(8);
                    int n_tokens = llama_tokenize(vocab, formatted_num, strlen(formatted_num),
                                                  tokens.data(), tokens.size(),
                                                  false, false);
                    
                    if (n_tokens > 0) {
                        tokens.resize(n_tokens);
                        
                        // Use first token (should be the main number token)
                        llama_token token_to_use = tokens[0];
                        
                        // Verify it's unique
                        if (std::find(temp_tokens.begin(), temp_tokens.end(), token_to_use) != temp_tokens.end()) {
                            all_unique = false;
                            printf("  Format leads to duplicate tokens for number %d\n", i);
                            break;
                        }
                        
                        temp_tokens.push_back(token_to_use);
                        
                        char token_text[32] = {0};
                        llama_token_to_piece(vocab, token_to_use, token_text, sizeof(token_text), 0, true);
                        printf("  Number %d -> token %d ('%s')\n", i, (int)token_to_use, token_text);
                    } else {
                        all_unique = false;
                        fprintf(stderr, "  Failed to tokenize number %d with format %s\n", i, format.c_str());
                        break;
                    }
                }
                
                if (all_unique && temp_tokens.size() == 10) {
                    success = true;
                    printf("Found alternative format with unique tokens: %s\n", format.c_str());
                    break;
                }
            }
            
            // If all formats failed, use a completely different approach: spell out numbers
            if (!success) {
                printf("All formats failed. Using spelled out numbers.\n");
                
                const char* spelled_numbers[] = {
                    "one", "two", "three", "four", "five", 
                    "six", "seven", "eight", "nine", "ten"
                };
                
                temp_tokens.clear();
                
                for (int i = 0; i < 10; i++) {
                    // Try with space prefix for better tokenization
                    std::string spelled = " ";
                    spelled += spelled_numbers[i];
                    
                    std::vector<llama_token> tokens(4);
                    int n_tokens = llama_tokenize(vocab, spelled.c_str(), spelled.length(),
                                                 tokens.data(), tokens.size(),
                                                 false, false);
                    
                    if (n_tokens > 0) {
                        tokens.resize(n_tokens);
                        temp_tokens.push_back(tokens[0]);
                        
                        char token_text[32] = {0};
                        llama_token_to_piece(vocab, tokens[0], token_text, sizeof(token_text), 0, true);
                        printf("Spelled number %d ('%s') -> token %d ('%s')\n", 
                               i+1, spelled_numbers[i], (int)tokens[0], token_text);
                    } else {
                        fprintf(stderr, "Failed to tokenize spelled number %d ('%s')\n", 
                                i+1, spelled_numbers[i]);
                    }
                }
                
                if (temp_tokens.size() > 0) {
                    success = true;
                }
            }
            
            if (success) {
                // Use the tokens we found with our alternative approaches
                number_tokens = std::move(temp_tokens);
            } else {
                // Last resort: use digit characters
                printf("All approaches failed. Using digit characters as a last resort.\n");
                
                for (int i = 1; i <= 10; i++) {
                    // Just use the digit character
                    char digit = '0' + (i % 10);
                    std::string digit_str(1, digit);
                    
                    std::vector<llama_token> tokens(4);
                    int n_tokens = llama_tokenize(vocab, digit_str.c_str(), digit_str.length(),
                                                tokens.data(), tokens.size(),
                                                false, false);
                    
                    if (n_tokens > 0) {
                        tokens.resize(n_tokens);
                        number_tokens.push_back(tokens[0]);
                        
                        char token_text[32] = {0};
                        llama_token_to_piece(vocab, tokens[0], token_text, sizeof(token_text), 0, true);
                        printf("Digit %d -> token %d ('%s')\n", i, (int)tokens[0], token_text);
                    }
                }
            }
        } else {
            // Use the tokens we found with the working format
            number_tokens = std::move(temp_tokens);
        }
    }
    
    // Print the tokenized content
    print_tokens(vocab, gettysburg_tokens, "Gettysburg");
    print_tokens(vocab, number_tokens, "Numbers");
    
    // Get token embeddings using our new API function
    std::vector<std::vector<float>> gettysburg_embeddings;
    for (size_t i = 0; i < gettysburg_tokens.size(); i++) {
        llama_token token = gettysburg_tokens[i];
        
        char token_text[32] = {0};
        llama_token_to_piece(vocab, token, token_text, sizeof(token_text), 0, true);
        
        printf("Getting embedding for Gettysburg token %zu: %d ('%s')\n", i, (int)token, token_text);
        
        float* embd = llama_token_get_embedding(model, token);
        if (embd) {
            printf("Successfully got embedding for token %d\n", (int)token);
            gettysburg_embeddings.push_back(std::vector<float>(embd, embd + n_embd));
        } else {
            fprintf(stderr, "Failed to get embedding for token %d\n", (int)token);
        }
    }
    
    std::vector<std::vector<float>> number_embeddings;
    for (size_t i = 0; i < number_tokens.size(); i++) {
        llama_token token = number_tokens[i];
        
        char token_text[32] = {0};
        llama_token_to_piece(vocab, token, token_text, sizeof(token_text), 0, true);
        
        printf("Getting embedding for number token %zu: %d ('%s')\n", i, (int)token, token_text);
        
        float* embd = llama_token_get_embedding(model, token);
        if (embd) {
            printf("Successfully got embedding for token %d\n", (int)token);
            number_embeddings.push_back(std::vector<float>(embd, embd + n_embd));
        } else {
            fprintf(stderr, "Failed to get embedding for token %d\n", (int)token);
        }
    }
    
    printf("Got %zu Gettysburg embeddings and %zu number embeddings\n", 
           gettysburg_embeddings.size(), number_embeddings.size());
    
    // Check if we have any embeddings before proceeding
    if (gettysburg_embeddings.empty() && number_embeddings.empty()) {
        fprintf(stderr, "No embeddings available. Cannot continue.\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    
    // Create probabilities for the mixture
    float weight_gettysburg = 0.5f;
    float weight_numbers = 0.5f;
    
    // Adjust weights if either set is empty
    if (gettysburg_embeddings.empty()) {
        weight_gettysburg = 0.0f;
        weight_numbers = 1.0f;
    } else if (number_embeddings.empty()) {
        weight_gettysburg = 1.0f;
        weight_numbers = 0.0f;
    }
    
    std::vector<float> gettysburg_weights;
    std::vector<float> number_weights;
    
    if (!gettysburg_embeddings.empty()) {
        gettysburg_weights.resize(gettysburg_embeddings.size(), weight_gettysburg / gettysburg_embeddings.size());
    }
    
    if (!number_embeddings.empty()) {
        number_weights.resize(number_embeddings.size(), weight_numbers / number_embeddings.size());
    }
    
    // Combine all embeddings and weights into single vectors
    std::vector<std::vector<float>> all_embeddings;
    std::vector<float> all_weights;
    
    all_embeddings.insert(all_embeddings.end(), gettysburg_embeddings.begin(), gettysburg_embeddings.end());
    all_embeddings.insert(all_embeddings.end(), number_embeddings.begin(), number_embeddings.end());
    
    all_weights.insert(all_weights.end(), gettysburg_weights.begin(), gettysburg_weights.end());
    all_weights.insert(all_weights.end(), number_weights.begin(), number_weights.end());
    
    printf("Creating mixed embedding with %zu source embeddings\n", all_embeddings.size());
    
    // Create the mixed embedding
    std::vector<float> mixed_embedding = create_weighted_embedding(all_embeddings, all_weights);
    
    // Verify the mixed embedding
    printf("Mixed embedding created with %zu elements\n", mixed_embedding.size());
    
    // Print a few values to verify
    printf("Mixed embedding sample values:\n");
    for (int i = 0; i < std::min(5, (int)mixed_embedding.size()); i++) {
        printf("  [%d]: %f\n", i, mixed_embedding[i]);
    }
    
    // Create batch with the mixed embedding
    printf("Initializing batch with embedding size %d\n", n_embd);
    
    // Make sure context params have embeddings and logits_all enabled
    ctx_params.embeddings = true;
    ctx_params.logits_all = true;
    
    // Ensure our context is set up for embeddings 
    llama_set_embeddings(ctx, true);
    
    // We need to create the batch properly to get logits
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    
    // Copy mixed embedding to batch
    if (mixed_embedding.size() != (size_t)n_embd) {
        fprintf(stderr, "Error: Mixed embedding size (%zu) doesn't match model's embedding size (%d)\n", 
                mixed_embedding.size(), n_embd);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("Copying embedding to batch...\n");
    memcpy(batch.embd, mixed_embedding.data(), n_embd * sizeof(float));
    
    batch.n_tokens = 1;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;  // Explicitly request logits for this token
    
    printf("Running inference with mixed embedding...\n");
    
    // Run inference with the mixed embedding
    int decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
        fprintf(stderr, "Failed to decode (error code: %d)\n", decode_result);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("Inference successful!\n");
    
    // Force synchronization to ensure all computations are complete
    llama_synchronize(ctx);
    printf("Context synchronized\n");
    
    // Safely get and process the output logits
    printf("Getting logits from context...\n");
    float* logits = llama_get_logits(ctx);
    if (!logits) {
        fprintf(stderr, "Failed to get logits from context\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    
    const int n_vocab = llama_vocab_n_tokens(vocab);
    printf("Processing logits for vocabulary size: %d\n", n_vocab);
    
    // Validate n_vocab is reasonable
    if (n_vocab <= 0 || n_vocab > 1000000) {  // Sanity check
        fprintf(stderr, "Invalid vocabulary size: %d\n", n_vocab);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    
    // Convert logits to probabilities
    std::vector<std::pair<float, llama_token>> token_probs;
    token_probs.reserve(n_vocab);
    
    // Find max logit for numerical stability (with bounds checking)
    printf("Finding maximum logit...\n");
    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; i++) {
        if (!std::isnan(logits[i]) && !std::isinf(logits[i]) && logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    if (std::isinf(max_logit) || std::isnan(max_logit)) {
        fprintf(stderr, "Invalid maximum logit value: %f\n", max_logit);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    
    printf("Max logit: %f\n", max_logit);
    
    // Compute softmax for probabilities with extra checks
    printf("Computing softmax...\n");
    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        // Skip NaN or Inf values
        if (std::isnan(logits[i]) || std::isinf(logits[i])) {
            token_probs.push_back({0.0f, i});
            continue;
        }
        
        float p = expf(logits[i] - max_logit);
        if (std::isnan(p) || std::isinf(p)) {
            p = 0.0f;
        }
        token_probs.push_back({p, i});
        sum_exp += p;
    }
    
    if (sum_exp <= 0.0f || std::isnan(sum_exp) || std::isinf(sum_exp)) {
        fprintf(stderr, "Invalid sum of exponentiated logits: %f\n", sum_exp);
        sum_exp = 1.0f;  // Prevent division by zero
    }
    
    // Normalize probabilities
    printf("Normalizing probabilities...\n");
    for (auto& tp : token_probs) {
        tp.first /= sum_exp;
        if (std::isnan(tp.first) || std::isinf(tp.first)) {
            tp.first = 0.0f;
        }
    }
    
    // Sort by probability in descending order
    printf("Sorting tokens by probability...\n");
    std::sort(token_probs.begin(), token_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Print top tokens from the output distribution
    printf("\nTop predicted tokens:\n");
    
    // Determine how many tokens to show (with bounds checking)
    int show_tokens = std::min(20, (int)token_probs.size());
    printf("Showing top %d tokens out of %zu total\n", show_tokens, token_probs.size());
    
    for (int i = 0; i < show_tokens; i++) {
        llama_token token_id = token_probs[i].second;
        float prob = token_probs[i].first * 100.0f; // Convert to percentage
        
        // Validate token ID
        if (token_id < 0 || token_id >= n_vocab) {
            fprintf(stderr, "Invalid token ID: %d\n", (int)token_id);
            continue;
        }
        
        // Get token text with extra safety
        char token_text[64] = {0};
        int len = llama_token_to_piece(vocab, token_id, token_text, sizeof(token_text) - 1, 0, true);
        if (len < 0) {
            strcpy(token_text, "<?>");
        }
        token_text[sizeof(token_text) - 1] = '\0';  // Ensure null termination
        
        printf("%2d. Token %6d (%-10s): %.2f%%\n", 
               i+1, (int)token_id, token_text, prob);
    }
    
    // Get expected continuations with improved error handling
    printf("\nGetting expected continuations for analysis...\n");
    
    std::vector<llama_token> expected_gettysburg_continuations;
    {
        std::string next_part = "on this continent";
        printf("Tokenizing expected Gettysburg continuation: '%s'\n", next_part.c_str());
        
        std::vector<llama_token> tokens(10);
        int n_tokens = llama_tokenize(vocab, next_part.c_str(), next_part.length(), 
                                      tokens.data(), tokens.size(), 
                                      false, false);
        
        if (n_tokens > 0) {
            tokens.resize(n_tokens);
            expected_gettysburg_continuations = tokens;
            
            printf("Tokenized into %d tokens:\n", n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                char token_text[64] = {0};
                llama_token_to_piece(vocab, tokens[i], token_text, sizeof(token_text) - 1, 0, true);
                printf("  Token %d: %d ('%s')\n", i, (int)tokens[i], token_text);
            }
        } else {
            fprintf(stderr, "Failed to tokenize Gettysburg continuation\n");
        }
    }
    
    // Use the same number format discovered earlier for consistency
    std::vector<llama_token> expected_number_continuations;
    printf("Tokenizing expected number continuations (11-15)...\n");
    
    for (int i = 11; i <= 15; i++) {
        // If we're using a format from earlier, use the same one
        std::string formatted_num;
        const char* formats[] = {
            "Number %d", "Count to %d", "%d.", "%d,", "(%d)", "=%d="
        };
        
        char buffer[64];
        if (number_tokens.size() >= 10) {  // If we found a good format earlier
            const char* format = formats[0];  // Default
            
            // For simplicity, use the first format (or adapt based on the tokens we found)
            snprintf(buffer, sizeof(buffer), format, i);
            formatted_num = buffer;
        } else {
            // Fallback to simple number
            formatted_num = std::to_string(i);
        }
        
        printf("Attempting to tokenize number: '%s'\n", formatted_num.c_str());
        
        std::vector<llama_token> tokens(4);
        int n_tokens = llama_tokenize(vocab, formatted_num.c_str(), formatted_num.length(), 
                                      tokens.data(), tokens.size(), 
                                      false, false);
        
        if (n_tokens > 0) {
            tokens.resize(n_tokens);
            expected_number_continuations.push_back(tokens[0]);
            
            char token_text[64] = {0};
            llama_token_to_piece(vocab, tokens[0], token_text, sizeof(token_text) - 1, 0, true);
            printf("  Tokenized number %d -> token %d ('%s')\n", 
                   i, (int)tokens[0], token_text);
        } else {
            fprintf(stderr, "Failed to tokenize number %d\n", i);
        }
    }
    
    // Print expected continuations and their ranks
    printf("\nExpected Gettysburg continuations:\n");
    
    if (expected_gettysburg_continuations.empty()) {
        printf("  No expected Gettysburg continuations available\n");
    }
    
    for (auto token : expected_gettysburg_continuations) {
        // Validate token
        if (token < 0 || token >= n_vocab) {
            fprintf(stderr, "Invalid token ID: %d\n", (int)token);
            continue;
        }
        
        // Get token text safely
        char token_text[64] = {0};
        int len = llama_token_to_piece(vocab, token, token_text, sizeof(token_text) - 1, 0, true);
        if (len < 0) {
            strcpy(token_text, "<?>");
        }
        token_text[sizeof(token_text) - 1] = '\0';
        
        // Find this token in the predictions
        auto it = std::find_if(token_probs.begin(), token_probs.end(), 
                             [token](const auto& tp) { return tp.second == token; });
        
        if (it != token_probs.end()) {
            int rank = std::distance(token_probs.begin(), it) + 1;
            float prob = it->first * 100.0f;
            printf("  Token %6d (%-10s): rank %3d, prob %.2f%%\n", 
                   (int)token, token_text, rank, prob);
        } else {
            printf("  Token %6d (%-10s): not in top predictions\n", 
                   (int)token, token_text);
        }
    }
    
    printf("\nExpected number continuations:\n");
    
    if (expected_number_continuations.empty()) {
        printf("  No expected number continuations available\n");
    }
    
    for (auto token : expected_number_continuations) {
        // Validate token
        if (token < 0 || token >= n_vocab) {
            fprintf(stderr, "Invalid token ID: %d\n", (int)token);
            continue;
        }
        
        // Get token text safely
        char token_text[64] = {0};
        int len = llama_token_to_piece(vocab, token, token_text, sizeof(token_text) - 1, 0, true);
        if (len < 0) {
            strcpy(token_text, "<?>");
        }
        token_text[sizeof(token_text) - 1] = '\0';
        
        // Find this token in the predictions
        auto it = std::find_if(token_probs.begin(), token_probs.end(), 
                             [token](const auto& tp) { return tp.second == token; });
        
        if (it != token_probs.end()) {
            int rank = std::distance(token_probs.begin(), it) + 1;
            float prob = it->first * 100.0f;
            printf("  Token %6d (%-10s): rank %3d, prob %.2f%%\n", 
                   (int)token, token_text, rank, prob);
        } else {
            printf("  Token %6d (%-10s): not in top predictions\n", 
                   (int)token, token_text);
        }
    }
    
    // Clean up
    printf("\nCleaning up resources...\n");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    printf("Done!\n");
    
    return 0;
}
