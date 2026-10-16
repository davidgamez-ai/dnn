import { AdamW } from "./AdamW.js";
import { crossEntropyLoss, oneHot } from "./Trainer.js";
import { XORNetwork } from "./XORNetwork.js";
/**
 * Trains an XORNetwork on the XOR truth table.
 *
 * Full-batch training
 * -------------------
 * The digit Trainer splits its data into training, validation and test sets,
 * shuffles it, and processes it in mini-batches. None of that is needed here:
 *
 *   - There are only four examples, and they are every possible input, so
 *     there is nothing to hold back for validation or testing. The network
 *     is judged on the same four examples it learns from.
 *   - All four fit easily in a single batch, so each weight update uses the
 *     exact average gradient over the whole data set rather than an estimate
 *     from a sample. This is called full-batch gradient descent.
 *   - With one batch per epoch, the order of the examples does not affect
 *     the averaged gradient, so there is no need to shuffle.
 *
 * As a result, a batch and an epoch are the same thing, so only the epoch
 * loss is logged.
 */
export class XORTrainer {
    _hyperparameters;
    _network;
    _optimizer;
    /**
     * Creates the optimizer that will update the network's weights.
     *
     * @param hyperparameters Settings for training: epochs and the AdamW
     * settings. batchSize and dataSplit are ignored (see the class comment).
     * The default learningRate of 0.001 is slow for XOR; 0.01 works well.
     * @param network The network to train. It is trained from its current
     * weights, which are the random starting weights if it is new.
     */
    constructor(hyperparameters, network) {
        this._hyperparameters = hyperparameters;
        this._network = network;
        this._optimizer = new AdamW(hyperparameters);
    }
    /**
     * Trains the network for the number of epochs in the hyperparameters.
     * Each epoch is a single batch of all four examples:
     *
     *   For each of the four examples:
     *     1. Feed the inputs forward.
     *     2. Measure the cross-entropy loss against the answer.
     *     3. Backpropagate, so both layers have their errors for this example.
     *     4. Calculate the gradients, averaging them over the four examples.
     *   Then:
     *     5. The AdamW optimizer updates the weights and biases once, using
     *        the averaged gradients.
     *
     * The average loss over the four examples is logged every logInterval
     * epochs, and after the last epoch. A network guessing 50/50 scores
     * -log(0.5), about 0.69; as it learns XOR the loss falls towards 0.
     *
     * Calling train() again continues from the current weights, and the
     * optimizer keeps its running averages.
     *
     * @param logInterval How many epochs between loss reports. XOR needs
     * thousands of epochs, so logging every one would be hard to read.
     * @throws RangeError if logInterval is not a positive integer.
     */
    train(logInterval = 100) {
        if (!Number.isInteger(logInterval) || logInterval < 1) {
            throw new RangeError(`logInterval must be a positive integer, got ${logInterval}`);
        }
        const epochs = this._hyperparameters.epochs;
        for (let epoch = 1; epoch <= epochs; epoch++) {
            let lossTotal = 0;
            XORNetwork.EXAMPLES.forEach((example, index) => {
                // Step 1: forward pass.
                const probabilities = this._network.update(example.inputs);
                // Step 2: loss.
                lossTotal += crossEntropyLoss(probabilities, example.answer);
                // Step 3: backpropagate the errors from the desired output.
                this._network.backpropagate(oneHot(example.answer, XORNetwork.NUM_CLASSES));
                // Step 4: gradients. The first example replaces the previous
                // epoch's gradients; the rest are averaged in.
                this._network.calculateGradients(index > 0);
            });
            // Step 5: one weight update per epoch, using the gradients
            // averaged over all four examples.
            this._network.updateWeights(this._optimizer);
            if (epoch % logInterval === 0 || epoch === epochs) {
                const loss = lossTotal / XORNetwork.EXAMPLES.length;
                console.log(`Epoch ${epoch}/${epochs}: average loss ${loss.toFixed(4)}`);
            }
        }
    }
    /** The network being trained. */
    get network() {
        return this._network;
    }
}
//# sourceMappingURL=XORTrainer.js.map