/** Switches that control debug log output in different parts of the model */
const DEBUG = {
    /** Switches on log output for image loading */
    IMAGE_LOAD: false,

    /** Switches on log output of the output layer's logits (values before softmax) */
    OUTPUT_LOGITS: false,

    /** Switches on log output of each output neuron's error signal during training */
    OUTPUT_ERROR_SIGNAL: true,

};

export default DEBUG;