/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

#include "clip.hpp"
#include "common/math.hpp"
#include "common/tensors.hpp"
#include "zmq.hpp"
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

// #include "queue.hpp"

const char *output_layer_name = "clip_resnet_50/conv59"; // there are 62 params

zmq::context_t zmq_context;
zmq::socket_t zmq_publisher;
zmq::socket_t zmq_subscriber;

bool new_prompt = false;
bool initialization_done = false; // Flag for initialization
bool first_running = true;
bool done_prompt = false;
std::mutex mtx;  // Mutex for synchronization
std::condition_variable cv;  // Condition variable for waiting

std::queue<std::vector<float>> m_image_embedding_queue;
std::queue<std::vector<std::vector<float>>> m_text_embedding_queue;
std::vector<std::string> prompts;
std::vector<std::vector<float>> embeddings;

std::vector<float> probs;


float logit_scale_1 = std::exp(4.60517); 


std::mutex image_queue_mutex;  // Mutex for the image embedding queue
std::mutex push_image_queue_mutex;  // Mutex for the image embedding queue
std::mutex text_queue_mutex;   // Mutex for the text embedding queue
std::mutex new_prompt_mutex; // Mutex for thread-safe access to new_prompt
std::mutex first_running_mutex; // Mutex for synchronization with first_running
std::mutex initialization_mutex;  // Mutex for initialization

std::thread probs_thread;

using json = nlohmann::json;


void init_zmq_publisher(const std::string &address) {
    try {
        // Create a new publisher socket with the context
        zmq_publisher = zmq::socket_t(zmq_context, zmq::socket_type::pub);

        // Bind the publisher socket to the provided address
        zmq_publisher.bind(address);

        // std::cout << "Publisher socket successfully bound to: " << address << std::endl;
    } catch (const zmq::error_t &e) {
        std::cerr << "Error initializing ZMQ publisher: " << e.what() << std::endl;
    }
}


/**
 * @brief Initialize the ZeroMQ SUB socket for subscribing to messages.
 * 
 * @param address The address to connect the SUB socket to.
 */
void init_zmq_subscriber(const std::string &address) {
    zmq_subscriber = zmq::socket_t(zmq_context, ZMQ_SUB);
    zmq_subscriber.connect(address);
    zmq_subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);  // Subscribe to all messages
}

/**
 * @brief Receive and process messages from the publisher using ZeroMQ SUB socket.
 * 
 * @param subscriber The ZeroMQ subscriber socket.
 */
void receive_messages(zmq::socket_t &zmq_subscriber) {
    while (true) {
        zmq::message_t zmq_msg;
        zmq_subscriber.recv(zmq_msg, zmq::recv_flags::none);

        std::string message_str(static_cast<char*>(zmq_msg.data()), zmq_msg.size());
        json received_json = json::parse(message_str);

        if (received_json.contains("prompts") && received_json["prompts"].is_array()) {
            prompts.clear(); 

            for (const auto& item : received_json["prompts"]) {
                if (item.is_null()) {
                    prompts.push_back(""); 
                } else {
                    prompts.push_back(item.get<std::string>());
                }
            }
        } else {
            std::cerr << "'prompts' field is missing or not an array" << std::endl;
        }

        // Parse the embeddings
        if (received_json.contains("embedding") && received_json["embedding"].is_array()) {
            embeddings.clear(); // Clear the embeddings vector
            for (const auto& item : received_json["embedding"]) {
                if (item.is_null()) {
                    embeddings.push_back({}); 
                } else {
                    embeddings.push_back(item.get<std::vector<float>>()); 
                }
            }
        } else {
            std::cerr << "'embedding' field is missing or not an array" << std::endl;
        }

        {
            std::lock_guard<std::mutex> lock(first_running_mutex);
            if (first_running) {
                first_running = false;
            } else {
                std::lock_guard<std::mutex> prompt_lock(new_prompt_mutex);
                new_prompt = true;
            }
        }

        std::lock_guard<std::mutex> lock(text_queue_mutex);  // Lock before pushing
        m_text_embedding_queue.push(embeddings);
    }
}

void send_probs(const std::vector<float>& probs) {
    size_t size = probs.size() * sizeof(float);
    zmq::message_t message(size);
    std::memcpy(message.data(), probs.data(), size);
    zmq_publisher.send(message, zmq::send_flags::none);
}

// Softmax function
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exp_logits(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    float sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == 0.0f) {
            continue;
        }
        exp_logits[i] = std::exp(logits[i] - max_logit);  // Stability improvement
        sum_exp += exp_logits[i];
    }
    
    for (size_t i = 0; i < exp_logits.size(); ++i) {
        exp_logits[i] /= sum_exp;
    }
    
    return exp_logits;
}

void normalize(std::vector<float>& vec) {
    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));

    if (norm != 0.0f) {
        std::transform(vec.begin(), vec.end(), vec.begin(), [norm](float v) { return v / norm; });
    }
}

void normalize_vectors(std::vector<std::vector<float>>& data) {
    for (auto& vec : data) {
        float norm = std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0f, [](float sum, float val) {
            return sum + val * val;
        }));

        if (norm == 0.0f) {
            continue;
        }

        for (auto& val : vec) {
            val /= norm;
        }
    }
}

std::vector<float> custom_dot_product(const std::vector<float>& A, const std::vector<std::vector<float>>& B) {
    std::vector<float> result(B.size());

    for (std::size_t i = 0; i < B.size(); ++i) {
        if(B[i].size() == 0){
            result[i] = 0.0f;
        }
        else{
            result[i] = std::inner_product(A.begin(), A.end(), B[i].begin(), 0.0f);
            result[i] = result[i] * logit_scale_1;
        }
    }

    return result;
}

void get_and_send_probs(std::vector<std::vector<float>>& text_embeddings) {
    while (true) {
        {
            std::unique_lock<std::mutex> image_lock(image_queue_mutex); 

            if (!m_image_embedding_queue.empty()) {
                std::vector<float> image_embeddings = m_image_embedding_queue.front();
                m_image_embedding_queue.pop();

                normalize(image_embeddings);
                normalize_vectors(text_embeddings); 

                std::vector<float> dot_product_result = custom_dot_product(image_embeddings, text_embeddings);

                probs = softmax(dot_product_result);

                send_probs(probs);
            }
        } 

        {
            std::lock_guard<std::mutex> prompt_lock(new_prompt_mutex); 
            if (new_prompt) {
                new_prompt = false;
                break;
            }
        }  
    }
}

/**
 * @brief calculate the probabilities and send them
 *
 * @param roi 
 */
void create_and_send_probs() {
    while (true) {
        std::unique_lock<std::mutex> lock(text_queue_mutex);  // Lock mutex before accessing the text queue

        if (!m_text_embedding_queue.empty()) {
            std::vector<std::vector<float>> text_embeddings = m_text_embedding_queue.front();
            m_text_embedding_queue.pop();
            
            get_and_send_probs(text_embeddings);
        } 
        lock.unlock();  
    }
    std::cout << "out from create_and_send_probs" << std::endl;
}

/**
 * @brief Get the image embedding and push it to the image embedding queue.
 *
 * @param roi 
 */
void get_image_embedding(HailoROIPtr roi){
    HailoTensorPtr tensor = roi->get_tensor("clip_resnet_50/conv59");
    if(tensor) {
        std::unique_lock<std::mutex> lock(image_queue_mutex);
        uint8_t *data_ptr = tensor->data();
        size_t data_size = tensor->size();

        std::vector<float> dequantized_data(data_size);
        // Dequantize the tensor data
        for (size_t i = 0; i < data_size; ++i) {
            dequantized_data[i] = tensor->fix_scale(data_ptr[i]);
        }

        // push the tensor data
        m_image_embedding_queue.push(dequantized_data);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lock.unlock();
    }
}

void clip(HailoROIPtr roi){
    std::lock_guard<std::mutex> lock(initialization_mutex);

    // if is the first runnig, we need to init the zmq and the threads
    if(!initialization_done){
        init_zmq_publisher("tcp://10.0.0.1:7000");
        init_zmq_subscriber("tcp://10.0.0.2:5555");

        std::thread subscriber_thread(receive_messages, std::ref(zmq_subscriber)); 
        std::thread probs_thread(create_and_send_probs);  

        subscriber_thread.detach();
        probs_thread.detach();
        initialization_done = true;
    }

    get_image_embedding(roi);

    std::shared_ptr<HailoDetection> detection = std::dynamic_pointer_cast<HailoDetection>(roi);

    if(detection){
        // auto max_prob = std::max_element(probs.begin(), probs.end());
        // int index = std::distance(probs.begin(), max_prob);
        // if(prompts.size() > 0){
        //     // detection -> set_label(prompts[index]);
        //     // std::cout << "before" << std::endl;
        //     // std::cout << *max_prob << std::endl;
        //     // std::cout << "after" << std::endl;
        //     // std::cout << *max_prob * 100 << std::endl;
        //     // detection -> set_confidence(*max_prob);
        // }
    }

}


void clip_resnet_50_nv12(HailoROIPtr roi) {
    output_layer_name = "clip_resnet_50/conv59";
    clip(roi);
}
