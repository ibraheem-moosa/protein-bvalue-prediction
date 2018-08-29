# protein-bvalue-prediction
Predict protein B value from sequence.

To run this first get the rnn input files from [here](https://drive.google.com/open?id=1URxykv0RfgC3f0XJLteZrQ8lwTILgaZZ).

Go to src directory.
Then run

    python3 torch_lstm.py path/to/rnn_inputs protein_list.txt path/to/rnn/dir
    
Here the third parameter is a directory where the RNNs will be saved at each epoch.


# dependencies
- Pytorch (Download the gpu version from pytorch website)
- Scipy
- scikit-learn
- matplotlib
- Biopython (If you have to generate the inputs, hopefully you wont. The downloaded files should work.)

# generating the rnn inputs

Download filtered_data from [here](https://drive.google.com/open?id=1g-ma0oHXBayzkCyijQdbWhLXFc6SeiX8).

Go to src directory.
Then run

    python3 rnn_preprocessdata.py path/to/filtered_data path/to/rnn_inputs protein_list.txt y
