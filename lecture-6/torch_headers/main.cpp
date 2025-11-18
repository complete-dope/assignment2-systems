#include <torch/torch.h>
#include <iostream>

using namespace std;

int main() {
    cout << torch::rand({2,2}) << endl;
}

