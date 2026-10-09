import { HiddenLayer } from "./HiddenLayer.js";
import { Hyperparameters } from "./Hyperparameters.js";
import { OutputLayer } from "./OutputLayer.js";
import { setSeed } from "./Random.js";

/**
 * A neural network for classifying the 8x8 digit images:
 * 64 inputs -> hidden layer (64 neurons) -> hidden layer (32 neurons) -> output layer (10 neurons).
 */
export class Network {
    /** Number of inputs: one per pixel of a flattened 8x8 image. */
    static readonly INPUT_SIZE = 64;

    /** Number of outputs: one per digit class, 0-9. */
    static readonly NUM_CLASSES = 10;

    private readonly _hiddenLayer1: HiddenLayer;
    private readonly _hiddenLayer2: HiddenLayer;
    private readonly _outputLayer: OutputLayer;

    /**
     * @param hyperparameters Settings for the network, including the random seed used by build().
     */
    constructor(readonly hyperparameters: Hyperparameters) {
        // Each layer's input size is the number of neurons in the layer before it.
        this._hiddenLayer1 = new HiddenLayer(Network.INPUT_SIZE, 64);
        this._hiddenLayer2 = new HiddenLayer(this._hiddenLayer1.numNeurons, 32);
        this._outputLayer = new OutputLayer(this._hiddenLayer2.numNeurons, Network.NUM_CLASSES);
    }

    /**
     * Builds every layer, creating their weights and biases with random
     * starting values. Must be called before calculate().
     *
     * The random number generator is first reset to the hyperparameters'
     * randomSeed, so the same seed always gives the same starting weights.
     */
    build(): void {
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
    calculate(pixels: readonly number[]): number[] {
        const hidden1Output = this._hiddenLayer1.calculate(pixels);
        const hidden2Output = this._hiddenLayer2.calculate(hidden1Output);
        return this._outputLayer.calculate(hidden2Output);
    }
}
