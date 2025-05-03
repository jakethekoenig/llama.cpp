#include "common.h"
#include "llama.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

// Helper function to create a weighted average of embeddings
std::vector<float> create_weighted_embedding(
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
void print_tokens(const llama_vocab* vocab, const std::vector<llama_token>& tokens, const char* name) {
    printf("%s tokens (%zu):\n", name, tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        char token_text[32] = {0};
        llama_token_to_piece(vocab, tokens[i], token_text, sizeof(token_text), 0, true);
        printf("  %zu: %6d '%s'\n", i, (int)tokens[i], token_text);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    // Initialize llama.cpp
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    
    // Parse command line arguments
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    
    // Apply params
    model_params.n_gpu_layers = params.n_gpu_layers;
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.embeddings = true;  // Enable embeddings mode
    
    // Path to the model
    const char* model_path = params.model.c_str();
    
    printf("Loading model: %s\n", model_path);
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model from '%s'\n", model_path);
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
    
    // Create number sequence tokens
    std::vector<llama_token> number_tokens;
    {
        for (int i = 1; i <= 10; i++) {
            std::string num_str = std::to_string(i);
            std::vector<llama_token> tokens(4);
            int n_tokens = llama_tokenize(vocab, num_str.c_str(), num_str.length(), 
                                          tokens.data(), tokens.size(), 
                                          false, false);
            if (n_tokens > 0) {
                tokens.resize(n_tokens);
                number_tokens.push_back(tokens[0]);
            }
        }
    }
    
    // Print the tokenized content
    print_tokens(vocab, gettysburg_tokens, "Gettysburg");
    print_tokens(vocab, number_tokens, "Numbers");
    
    // Get token embeddings using our new API function
    std::vector<std::vector<float>> gettysburg_embeddings;
    for (llama_token token : gettysburg_tokens) {
        float* embd = llama_token_get_embedding(model, token);
        if (embd) {
            gettysburg_embeddings.push_back(std::vector<float>(embd, embd + n_embd));
        } else {
            fprintf(stderr, "Failed to get embedding for token %d\n", (int)token);
        }
    }
    
    std::vector<std::vector<float>> number_embeddings;
    for (llama_token token : number_tokens) {
        float* embd = llama_token_get_embedding(model, token);
        if (embd) {
            number_embeddings.push_back(std::vector<float>(embd, embd + n_embd));
        } else {
            fprintf(stderr, "Failed to get embedding for token %d\n", (int)token);
        }
    }
    
    printf("Got %zu Gettysburg embeddings and %zu number embeddings\n", 
           gettysburg_embeddings.size(), number_embeddings.size());
    
    // Create probabilities for the mixture
    // Use 0.5 for Gettysburg and 0.5 for numbers
    float weight_gettysburg = 0.5f;
    float weight_numbers = 0.5f;
    
    std::vector<float> gettysburg_weights(gettysburg_embeddings.size(), weight_gettysburg / gettysburg_embeddings.size());
    std::vector<float> number_weights(number_embeddings.size(), weight_numbers / number_embeddings.size());
    
    // Combine all embeddings and weights into single vectors
    std::vector<std::vector<float>> all_embeddings;
    std::vector<float> all_weights;
    
    all_embeddings.insert(all_embeddings.end(), gettysburg_embeddings.begin(), gettysburg_embeddings.end());
    all_embeddings.insert(all_embeddings.end(), number_embeddings.begin(), number_embeddings.end());
    
    all_weights.insert(all_weights.end(), gettysburg_weights.begin(), gettysburg_weights.end());
    all_weights.insert(all_weights.end(), number_weights.begin(), number_weights.end());
    
    // Create the mixed embedding
    std::vector<float> mixed_embedding = create_weighted_embedding(all_embeddings, all_weights);
    
    // Create batch with the mixed embedding
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    
    // Copy mixed embedding to batch
    memcpy(batch.embd, mixed_embedding.data(), n_embd * sizeof(float));
    batch.n_tokens = 1;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    
    printf("Running inference with mixed embedding...\n");
    
    // Run inference with the mixed embedding
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    // Get the output logits
    float* logits = llama_get_logits(ctx);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    
    // Convert logits to probabilities
    std::vector<std::pair<float, llama_token>> token_probs;
    token_probs.reserve(n_vocab);
    
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Compute softmax for probabilities
    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        float p = expf(logits[i] - max_logit);
        token_probs.push_back({p, i});
        sum_exp += p;
    }
    
    // Normalize probabilities
    for (auto& tp : token_probs) {
        tp.first /= sum_exp;
    }
    
    // Sort by probability in descending order
    std::sort(token_probs.begin(), token_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Print top tokens from the output distribution
    printf("\nTop predicted tokens:\n");
    for (int i = 0; i < 20 && i < (int)token_probs.size(); i++) {
        llama_token token_id = token_probs[i].second;
        float prob = token_probs[i].first * 100.0f; // Convert to percentage
        
        char token_text[32] = {0};
        llama_token_to_piece(vocab, token_id, token_text, sizeof(token_text), 0, true);
        
        printf("%2d. Token %6d (%-10s): %.2f%%\n", 
               i+1, (int)token_id, token_text, prob);
    }
    
    // Get expected continuations
    std::vector<llama_token> expected_gettysburg_continuations;
    {
        std::string next_part = "on this continent";
        std::vector<llama_token> tokens(10);
        int n_tokens = llama_tokenize(vocab, next_part.c_str(), next_part.length(), 
                                      tokens.data(), tokens.size(), 
                                      false, false);
        if (n_tokens > 0) {
            tokens.resize(n_tokens);
            expected_gettysburg_continuations = tokens;
        }
    }
    
    std::vector<llama_token> expected_number_continuations;
    for (int i = 11; i <= 15; i++) {
        std::string num_str = std::to_string(i);
        std::vector<llama_token> tokens(4);
        int n_tokens = llama_tokenize(vocab, num_str.c_str(), num_str.length(), 
                                      tokens.data(), tokens.size(), 
                                      false, false);
        if (n_tokens > 0) {
            tokens.resize(n_tokens);
            expected_number_continuations.push_back(tokens[0]);
        }
    }
    
    // Print expected continuations and their ranks
    printf("\nExpected Gettysburg continuations:\n");
    for (auto token : expected_gettysburg_continuations) {
        char token_text[32] = {0};
        llama_token_to_piece(vocab, token, token_text, sizeof(token_text), 0, true);
        
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
    for (auto token : expected_number_continuations) {
        char token_text[32] = {0};
        llama_token_to_piece(vocab, token, token_text, sizeof(token_text), 0, true);
        
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
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}
