/** Switches that control debug log output in different parts of the model */
const DEBUG = {
    /** Switches on log output for image loading */
    IMAGE_LOAD: false,
    /** Switches on log output of the output layer's logits (values before softmax) */
    OUTPUT_LOGITS: false,
    /** Switches on log output of each output neuron's error signal during training */
    OUTPUT_ERROR_SIGNAL: false,
    /** Switches on log output of each batch's average loss during training */
    BATCH_LOSS: false,
    /**
     * Switches on log output of every XORNetwork.update() call: the inputs,
     * each hidden neuron's activation, the output logits and the output
     * probabilities. Called four times per epoch during training, so this
     * produces a lot of output with many epochs.
     */
    XOR_UPDATE: true,
    /** Switches on log output of the XORNetwork's weights and biases when build() is called */
    XOR_WEIGHTS: true,
    /**
     * Switches on log output of every XORNetwork.backpropagate() call: the
     * target, each output neuron's error signal, the errors passed back to
     * the hidden layer and each hidden neuron's error signal. Called four
     * times per epoch during training, so this produces a lot of output with
     * many epochs.
     */
    XOR_ERROR_SIGNAL: true,
    /**
     * Switches on log output of every XORNetwork.calculateGradients() call:
     * the weight and bias gradients of both layers, and how many examples
     * they are averaged over. Called four times per epoch during training,
     * so this produces a lot of output with many epochs.
     */
    XOR_GRADIENTS: true,
};
export default DEBUG;
//# sourceMappingURL=Debug.js.map