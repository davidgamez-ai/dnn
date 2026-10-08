import { HiddenLayer } from "./HiddenLayer.js";
import { OutputLayer } from "./OutputLayer.js";
/**
 * A neural network for classifying the 8x8 digit images:
 * 64 inputs -> hidden layer (64 neurons) -> hidden layer (32 neurons) -> output layer (10 neurons).
 */
export class Network {
    /** Number of inputs: one per pixel of a flattened 8x8 image. */
    static INPUT_SIZE = 64;
    /** Number of outputs: one per digit class, 0-9. */
    static NUM_CLASSES = 10;
    _hiddenLayer1;
    _hiddenLayer2;
    _outputLayer;
    constructor() {
        // Each layer's input size is the number of neurons in the layer before it.
        this._hiddenLayer1 = new HiddenLayer(Network.INPUT_SIZE, 64);
        this._hiddenLayer2 = new HiddenLayer(this._hiddenLayer1.numNeurons, 32);
        this._outputLayer = new OutputLayer(this._hiddenLayer2.numNeurons, Network.NUM_CLASSES);
    }
    /**
     * Builds every layer, creating their weights and biases with random
     * starting values. Must be called before calculate().
     */
    build() {
        this._hiddenLayer1.build();
        this._hiddenLayer2.build();
        this._outputLayer.build();
    }
    /**
     * Feeds an image through the network and returns the probability of each digit.
     * The output of each layer becomes the input to the next.
     *
     * @param pixels The image's pixel values as a 1D array of INPUT_SIZE values, from ImageLoader.flatten().
     * @returns NUM_CLASSES probabilities, where element d is the probability that the image is digit d.
     * @throws RangeError if pixels does not have INPUT_SIZE values.
     * @throws Error if build() has not been called yet.
     */
    calculate(pixels) {
        const hidden1Output = this._hiddenLayer1.calculate(pixels);
        const hidden2Output = this._hiddenLayer2.calculate(hidden1Output);
        return this._outputLayer.calculate(hidden2Output);
    }
}
//# sourceMappingURL=Network.js.map