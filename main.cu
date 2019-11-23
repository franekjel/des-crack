#include <time.h>
#include <unistd.h>

#include "cpu_algorithm.h"
#include "device_algorithm.h"

__no_return__ void usage(char* pname)
{
    fprintf(stderr, "USAGE: %s [OPTIONS] - crack DES using CUDA\n\n\
Program can operate in two modes:\n\
1. Get encrypted and decrypted message and find key\n\
-e <64byte>     encrypted message to decrypt\n\
-d <64byte>     decrypted message\n\
64byte means 64-byte hex number, like 6c6f6e6b0da074c8 (16 digits, [0-9a-f])\n\
2. Generate key of given length and then crack it. Parameter:\n\
-s <int>        key size (1 - 64)\n\
There is also flags without parameters: -c (CPU) and -g (GPU) indicating which algorithm program should use (default GPU)\n\
Example:\n\
%s -e 6c6f6e6b0da074c8 -d 797d226c6f6a6b00 -c   will find key which encrypt second argument into first using CPU\n\
%s                                              with no parameters program will generate random data and try to break it using GPU\n\
%s -s 26                                        program will generate 26-bit key and then crack it (useful to measure performance)\n",
        pname, pname, pname, pname);
    exit(EXIT_FAILURE);
}

uint64_t rand64()
{
    uint64_t re = rand();
    re *= rand();
    re *= rand();
    re *= rand();
    return re;
}

int main(int argc, char* argv[])
{
    unsigned int s = 0;
    int c;
    int e = 0, d = 0, device = 0; //0 GPU; 1 CPU
    uint64_t encryped = 0;
    uint64_t key = 0;
    uint64_t decrypted = 0;

    while ((c = getopt(argc, argv, "-:cge:d:s:")) != -1) {
        switch (c) {
        case 'e': {
            sscanf(optarg, "%llx", &encryped);
            if (e)
                usage(argv[0]);
            e = 1;
            printf("Encrypted message: %lx\n", encryped);
        } break;
        case 'd': {
            sscanf(optarg, "%llx", &decrypted);
            if (d)
                usage(argv[0]);
            d = 1;
            printf("Decrypted message: %lx\n", decrypted);
        } break;
        case 's': {
            sscanf(optarg, "%d", &s);
            if (s < 1 || s > 64)
                usage(argv[0]);
        } break;
        case 'c': {
            device = 1;
        } break;
        case 'g': {
            device = 0;
        } break;
        case ':': {
            printf("Missing argument!\n");
            usage(argv[0]);
        }
        default: {
            usage(argv[0]);
        }
        }
    }

    if (e + d == 1) //there is d or e but not both
        usage(argv[0]);

    if (e && d) { //normal case, we want to find DES key
        if (device == 0)
            key = CUDACrackDES(encryped, decrypted);
        else
            key = CPUCrackDES(encryped, decrypted);
        printf("Found key: %lx\n", key);
        return 0;
    }

    //there is no parameters or only -s - we want to generate key and then crack it
    srand(1234);
    key = rand64();
    key >>= 64 - s;
    printf("Generated key: %lx\n", key);
    decrypted = rand64();
    printf("Generated message: %lx\n", decrypted);
    encryped = doDES(key, decrypted);
    printf("Message encrypted: %lx\n", encryped);
    unsigned long start = time(NULL);
    if (device == 0)
        key = CUDACrackDES(encryped, decrypted);
    else {
        key = CPUCrackDES(encryped, decrypted);
        printf("Time: %ld s\n", time(NULL) - start);
    }
    printf("Found key: %lx\n", key);

    return 0;
}
