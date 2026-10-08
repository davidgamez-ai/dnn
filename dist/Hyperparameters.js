/**
 * Settings that control how the neural network is built and trained.
 * Unlike the network's weights and biases, these are chosen before
 * training starts and are not learned from the data.
 */
export class Hyperparameters {
    _learningRate;
    _epochs;
    _batchSize;
    _hiddenLayerSizes;
    _dataSplit;
    /**
     * Creates a set of hyperparameters. Any value that is not supplied
     * takes its default.
     */
    constructor({ learningRate = 0.1, epochs = 50, batchSize = 32, hiddenLayerSizes = [32], dataSplit = { train: 0.7, validation: 0.15, test: 0.15 }, } = {}) {
        if (!(learningRate > 0)) {
            throw new RangeError(`learningRate must be greater than 0, got ${learningRate}`);
        }
        if (!Number.isInteger(epochs) || epochs < 1) {
            throw new RangeError(`epochs must be a positive integer, got ${epochs}`);
        }
        if (!Number.isInteger(batchSize) || batchSize < 1) {
            throw new RangeError(`batchSize must be a positive integer, got ${batchSize}`);
        }
        if (hiddenLayerSizes.some(size => !Number.isInteger(size) || size < 1)) {
            throw new RangeError(`hiddenLayerSizes must all be positive integers, got [${hiddenLayerSizes.join(", ")}]`);
        }
        const { train, validation, test } = dataSplit;
        if (!(train > 0) || !(validation >= 0) || !(test >= 0)) {
            throw new RangeError(`dataSplit.train must be greater than 0 and validation and test must not be negative, got ${train}/${validation}/${test}`);
        }
        // Allow for floating-point rounding, e.g. 0.7 + 0.15 + 0.15 is not exactly 1.
        if (Math.abs(train + validation + test - 1) > 1e-9) {
            throw new RangeError(`dataSplit fractions must add up to 1, got ${train}/${validation}/${test}`);
        }
        this._learningRate = learningRate;
        this._epochs = epochs;
        this._batchSize = batchSize;
        this._hiddenLayerSizes = [...hiddenLayerSizes];
        this._dataSplit = { train, validation, test };
    }
    /** Step size used when updating the weights after each batch. */
    get learningRate() {
        return this._learningRate;
    }
    /** Number of complete passes through the training data. */
    get epochs() {
        return this._epochs;
    }
    /** Number of training examples processed before the weights are updated. */
    get batchSize() {
        return this._batchSize;
    }
    /**
     * Number of neurons in each hidden layer, from input side to output side.
     * Returns a copy, so changing it does not affect these hyperparameters.
     */
    get hiddenLayerSizes() {
        return [...this._hiddenLayerSizes];
    }
    /**
     * How the dataset is divided into training, validation and test sets.
     * Returns a copy, so changing it does not affect these hyperparameters.
     */
    get dataSplit() {
        return { ...this._dataSplit };
    }
}
//# sourceMappingURL=Hyperparameters.js.map