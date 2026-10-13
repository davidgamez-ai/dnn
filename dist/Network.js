import { HiddenLayer } from "./HiddenLayer.js";
import { OutputLayer } from "./OutputLayer.js";
import { setSeed } from "./Random.js";
/**
 * A neural network for classifying the 8x8 digit images:
 * 64 inputs -> hidden layer (64 neurons) -> hidden layer (32 neurons) -> output layer (10 neurons).
 */
export class Network {
    hyperparameters;
    /** Number of inputs: one per pixel of a flattened 8x8 image. */
    static INPUT_SIZE = 64;
    /** Number of outputs: one per digit class, 0-9. */
    static NUM_CLASSES = 10;
    _hiddenLayer1;
    _hiddenLayer2;
    _outputLayer;
    /**
     * @param hyperparameters Settings for the network, including the random seed used by build().
     */
    constructor(hyperparameters) {
        this.hyperparameters = hyperparameters;
        // Each layer's input size is the number of neurons in the layer before it.
        this._hiddenLayer1 = new HiddenLayer(Network.INPUT_SIZE, 64);
        this._hiddenLayer2 = new HiddenLayer(this._hiddenLayer1.numNeurons, 32);
        this._outputLayer = new OutputLayer(this._hiddenLayer2.numNeurons, Network.NUM_CLASSES);
    }
    /**
     * Builds every layer, creating their weights and biases with random
     * starting values. Must be called before update().
     *
     * The random number generator is first reset to the hyperparameters'
     * randomSeed, so the same seed always gives the same starting weights.
     */
    build() {
        setSeed(this.hyperparameters.randomSeed);
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
    update(pixels) {
        const hidden1Output = this._hiddenLayer1.update(pixels);
        const hidden2Output = this._hiddenLayer2.update(hidden1Output);
        return this._outputLayer.update(hidden2Output);
    }
    /**
     * Works out every neuron's error for the most recent image passed to
     * update(), starting at the output layer and working backwards:
     *
     *     output layer   -> errors from comparing its output with the target
     *     hidden layer 2 -> errors passed back from the output layer
     *     hidden layer 1 -> errors passed back from hidden layer 2
     *
     * Afterwards each layer's errors property holds its up-to-date errors.
     *
     * @param target The desired output as a one-hot vector: 1 for the correct
     * digit and 0 for the rest, e.g. [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for the digit 3.
     * @throws RangeError if target is not a one-hot vector of NUM_CLASSES values.
     * @throws Error if update() has not been called yet.
     */
    backpropagate(target) {
        // Output layer: compare the output with the target.
        const outputErrors = this._outputLayer.calculateErrors(target);
        // Hidden layer 2: errors passed back through the output layer's weights.
        const errorsForHidden2 = this._outputLayer.calculatePreviousLayerErrors(outputErrors);
        const hidden2Errors = this._hiddenLayer2.calculateErrors(errorsForHidden2);
        // Hidden layer 1: errors passed back through hidden layer 2's weights.
        const errorsForHidden1 = this._hiddenLayer2.calculatePreviousLayerErrors(hidden2Errors);
        this._hiddenLayer1.calculateErrors(errorsForHidden1);
        // Hidden layer 1's inputs are the image pixels, which are not learned,
        // so there is nothing further back to pass errors to.
    }
    /**
     * Asks every layer to calculate the gradients of its weights and biases
     * from its stored errors and inputs. Afterwards each layer's
     * weightGradients and biasGradients properties are up to date.
     *
     * Call backpropagate() first, so every layer has errors for the current image.
     * Each layer's gradients depend only on its own errors and inputs, so the
     * order in which the layers are visited does not matter.
     *
     * @param batchAverage true to average the new gradients with each layer's
     * stored gradients (a running mean over the batch); false to replace them.
     * Use false for the first image of each batch and true for the rest.
     * @throws Error if update() or backpropagate() has not been called yet.
     */
    calculateGradients(batchAverage) {
        this._outputLayer.calculateGradients(this._outputLayer.errors, batchAverage);
        this._hiddenLayer2.calculateGradients(this._hiddenLayer2.errors, batchAverage);
        this._hiddenLayer1.calculateGradients(this._hiddenLayer1.errors, batchAverage);
    }
    /**
     * The output layer's errors (probability - target) from the most recent
     * backpropagate() call, one per digit.
     * @throws Error if backpropagate() has not been called yet.
     */
    get outputErrors() {
        return this._outputLayer.errors;
    }
}
//# sourceMappingURL=Network.js.map