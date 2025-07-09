#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <iostream>
#include <csignal>
#include <sstream>

const std::string PROMPT_TEXT_PREFIX = "<|im_start|>system\n你是一名专业AI助手，请用中文直接回答问题，回答要简洁明了，字数控制在200字以内<|im_end|>\n<|im_start|>user\n";
const std::string PROMPT_TEXT_POSTFIX = "<|im_end|>\n<|im_start|>assistant\n";
const std::string RESPONSE_TERMINATOR = "<|endoftext|>";

LLMHandle llmHandle = nullptr;

void exit_handler(int signal) {
    if (llmHandle != nullptr) {
        std::cout << "\n程序即将退出" << std::endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
    }
    exit(signal);
}

void callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    static std::string full_response;
    
    if (state == RKLLM_RUN_NORMAL) {
        full_response += result->text;
        printf("%s", result->text);
    }
    else if (state == RKLLM_RUN_FINISH) {
        printf("%s\n", RESPONSE_TERMINATOR.c_str());
        full_response.clear();
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);
    param.skip_special_token = true;

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret != 0) {
        std::cerr << "rkllm init failed" << std::endl;
        return -1;
    }
    printf("rkllm init success\n");

    while (true) {
        std::string question;
        std::cout << "\n";
        std::getline(std::cin, question);
        
        if (question == "exit" || question == "quit") {
            break;
        }

        // 使用前缀+用户问题+后缀构建完整提示
        std::string full_prompt = PROMPT_TEXT_PREFIX + question + PROMPT_TEXT_POSTFIX;
        
        RKLLMInput input;
        input.input_type = RKLLM_INPUT_PROMPT;
        input.prompt_input = full_prompt.c_str();

        RKLLMInferParam infer_param;
        memset(&infer_param, 0, sizeof(infer_param));
        infer_param.mode = RKLLM_INFER_GENERATE;

        rkllm_run(llmHandle, &input, &infer_param, NULL);
        
        // 等待输出完成
        usleep(100000); // 100ms延迟
    }
    
    rkllm_destroy(llmHandle);
    return 0;
}
