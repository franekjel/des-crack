#include "cpu_algorithm.h"
#include "device_algorithm.cu"

#include <time.h>
#include <unistd.h>

__no_return__ void usage(char* pname)
{
    fprintf(stderr, "USAGE: %s [OPTIONS] - crack DES using CUDA\n\n\
-e <64byte>: encrypted message to decrypt\n\
-k <64byte>: DES key\n\
-d <64byte>: decrypted message\n\
64byte means 64-byte hex number, like 6c6f6e6b0da074c8 (16 digits, [0-9a-f])\n\
In case of missing parameters program will generate random numbers\n\n\
Example:\n\
%s -e 6c6f6e6b0da074c8 -d 797d226c6f6a6b00    will find key which encrypt second argument into first\n\
%s    with no parameters program will generate random data and try to break it\n\
%s -k 12fa8335b83b82c2    program will generate random encrypted and decrypted message using key, and then try to crack they (useful to measure performance)",
        pname, pname, pname, pname);
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{

    int c;
    int e = 0, k = 0, d = 0;
    uint64_t encryped = 0;
    uint64_t key = 0;
    uint64_t decrypted = 0;

    while ((c = getopt(argc, argv, "-:e:k:d:")) != -1) {
        switch (c) {
        case 'e': {
            sscanf(optarg, "%lx", &encryped);
            if (e)
                usage(argv[0]);
            e = 1;
            printf("Encrypted message: %lx\n", encryped);
        } break;
        case 'k': {
            sscanf(optarg, "%lx", &key);
            if (k)
                usage(argv[0]);
            k = 1;
            printf("Key: %lx\n", key);
        } break;
        case 'd': {
            sscanf(optarg, "%lx", &decrypted);
            if (d)
                usage(argv[0]);
            d = 1;
            printf("Decrypted message: %lx\n", decrypted);
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

    if (e && d) { //normal case, we want to find DES key
        crackDES(encryped, decrypted);
        return 0;
    }

    if (!k) { //there is only e or d or nothing
        //generate key
        srand((unsigned)time(NULL));
        key = rand() * rand() % 1 > 56;
        if (e) //there is e but not d
            decrypted = doDES(key, encryped);
        if (d) //there is d but no e
            encryped = doDES(key, decrypted);
        if (!d && !e) { //there is no e nor d
            decrypted = (unsigned)rand() * (unsigned)rand();
            encryped = doDES(key, decrypted);
        }
        crackDES(encryped, decrypted);
    }

    return 0;
}
