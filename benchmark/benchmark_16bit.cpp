#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <string>
#include "charls/charls.hpp" 

// Read whole file
std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open " << path << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file1> <file2> ..." << std::endl;
        return 1;
    }

    double total_dec_ms = 0;
    double total_enc_ms = 0;
    size_t total_bytes = 0;
    
    std::cout << std::left << std::setw(30) << "File" 
              << " | " << std::setw(8) << "Dec(ms)" 
              << " | " << std::setw(8) << "Enc(ms)" 
              << " | " << std::setw(8) << "Size(KB)"
              << " | " << std::setw(4) << "Bps"
              << " | " << std::setw(4) << "Cmp" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (int i = 1; i < argc; ++i) {
        std::string path = argv[i];
        if (path.empty()) continue;
        

        auto encoded_data = read_file(path);
        if (encoded_data.empty()) continue;

        std::vector<uint8_t> encoded_stream;
        std::string filename = path.substr(path.find_last_of("/\\") + 1);

        // Check for PGM
        if (encoded_data.size() > 3 && encoded_data[0] == 'P' && encoded_data[1] == '5') {
             // Parse PGM
             size_t pos = 3; // Skip P5\n (or P5 )
             auto skip_ws = [&](size_t& p) {
                 while (p < encoded_data.size() && isspace(encoded_data[p])) {
                     if (encoded_data[p] == '#') { // Comment
                         while (p < encoded_data.size() && encoded_data[p] != '\n') p++;
                     }
                     p++;
                 }
             };
             
             auto read_int = [&](size_t& p) -> int {
                 skip_ws(p);
                 int val = 0;
                 while (p < encoded_data.size() && isdigit(encoded_data[p])) {
                     val = val * 10 + (encoded_data[p] - '0');
                     p++;
                 }
                 return val;
             };

             int width = read_int(pos);
             int height = read_int(pos);
             int max_val = read_int(pos); // 4095
             // Single whitespace after max_val
             if (pos < encoded_data.size() && isspace(encoded_data[pos])) pos++;

             // Convert Mono to RGB (3 components) to test Color SIMD
             std::vector<uint8_t> raw_pixels(encoded_data.begin() + pos, encoded_data.end());
             std::vector<uint8_t> rgb_pixels(width * height * 3 * 2); // 16-bit * 3
             // Input raw_pixels is 16-bit mono (2 bytes per pixel)
             const uint16_t* psrc = reinterpret_cast<const uint16_t*>(raw_pixels.data());
             uint16_t* pdst = reinterpret_cast<uint16_t*>(rgb_pixels.data());
             for(int k=0; k < width*height; ++k) {
                 uint16_t val = psrc[k]; // Endianness? Assuming passing through
                 if (max_val > 255) {
                      // Swap if needed? PGM is Big Endian. Host is Little.
                      // charls expects Little Endian for 16-bit usually (buffer).
                      // Let's swap to be safe for "value" operations like transform.
                      val = (val >> 8) | (val << 8);
                 }
                 pdst[3*k + 0] = val;
                 pdst[3*k + 1] = val;
                 pdst[3*k + 2] = val;
             }
             
             // Encode to JLS
             charls::jpegls_encoder encoder;
             encoder.frame_info({static_cast<uint32_t>(width), static_cast<uint32_t>(height), 
                                 max_val > 255 ? 16 : 8, 3}); // RGB
             encoder.color_transformation(charls::color_transformation::hp1);
             
             encoded_stream.resize(width * height * 6 + 1024);
             encoder.destination(encoded_stream);
             size_t size = encoder.encode(rgb_pixels);
             encoded_stream.resize(size);
        } else {
             encoded_stream = encoded_data;
        }

        if (encoded_stream.empty()) continue;

        // Skip custom header if present
        size_t offset = 0;
        if (encoded_stream.size() > 2 && !(encoded_stream[0] == 0xFF && encoded_stream[1] == 0xD8)) {
             // scan for FF D8
             for(size_t k=0; k < encoded_stream.size()-1; ++k) {
                 if (encoded_stream[k] == 0xFF && encoded_stream[k+1] == 0xD8) {
                     offset = k;
                     break;
                 }
             }
        }
        
        // Pass substring
        charls::jpegls_decoder decoder;
        // source takes buffer, size. We can pass pointer.
        decoder.source(encoded_stream.data() + offset, encoded_stream.size() - offset);
        try {
            decoder.read_header();
        } catch(const std::exception& e) {
             std::cerr << "Header read failed for " << path << ": " << e.what() << std::endl;
             continue;
        }

        std::vector<uint8_t> decoded_data(decoder.get_destination_size());
        
        auto start = std::chrono::high_resolution_clock::now();
        decoder.decode(decoded_data);
        auto end = std::chrono::high_resolution_clock::now();
        
        double dec_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        total_dec_ms += dec_ms;

        // --- ENCODE (Round trip) ---
        // Use same params
        charls::jpegls_encoder encoder;
        encoder.frame_info(decoder.frame_info());
        if (decoder.frame_info().component_count == 3) {
             // Forcing line interleave to match typical optimization path
             encoder.interleave_mode(charls::interleave_mode::line); 
        } else {
             encoder.interleave_mode(decoder.get_interleave_mode());
        }
        
        encoder.near_lossless(decoder.get_near_lossless()); 

        std::vector<uint8_t> re_encoded_data(decoded_data.size() * 2 + 1024);
        encoder.destination(re_encoded_data);
        
        start = std::chrono::high_resolution_clock::now();
        size_t encoded_size = encoder.encode(decoded_data);
        end = std::chrono::high_resolution_clock::now();
        
        double enc_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        total_enc_ms += enc_ms;
        total_bytes += decoded_data.size();
        
        std::cout << std::left << std::setw(30) << filename 
                  << " | " << std::setw(8) << dec_ms 
                  << " | " << std::setw(8) << enc_ms 
                  << " | " << std::setw(8) << (encoded_stream.size() / 1024)
                  << " | " << std::setw(4) << decoder.frame_info().bits_per_sample
                  << " | " << std::setw(4) << decoder.frame_info().component_count 
                  << " | " << decoder.frame_info().width << "x" << decoder.frame_info().height << std::endl;
    }
    
    std::cout << std::string(75, '-') << std::endl;
    std::cout << "Totals: Dec=" << total_dec_ms << "ms, Enc=" << total_enc_ms << "ms" << std::endl;
    double mps = (double)total_bytes / (total_dec_ms * 1000.0); // MegaBytes / sec (approx)
    std::cout << "Avg Dec Throughput: " << (total_bytes / 1024.0 / 1024.0) / (total_dec_ms / 1000.0) << " MB/s" << std::endl;
    
    return 0;
}
