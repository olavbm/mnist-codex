# Rules

Your goal is to implement a static MNIST classifier as *code* rather than using familiar learning techniques.

You may be tempted to take shortcuts by leveraging learning techniques. Do not do this.

 * Do not embed a parametric model or training set examples into the classifier code.
 * Keep the classifier small and looking like real code with properly named methods "eg detect_edge", loops, if statements, etc.
 * Do not do any learning (e.g. training regression models or trees) to produce the classifier code. No torch or sklearn.
 * Do not, under any circumstances, download the MNIST test set. Only look at the train set.
 * The classifier cannot be built at runtime from the training set. It must be static code that could run even if no train dataset were available. It is a stateless function that takes an image and returns a prediction.
 * There should be no fit() function, since that would imply learning when running the program.
 * Do not embed large tables of coefficients, as this starts to look like embedding a fit parametric model into the code.

You *may* try many ideas and see what happens to the accuracy. Your goal is to implement or remove functionality that behaves like interpretable code and classifies images correctly.

You are effectively hill climbing training accuracy using interpretable code, empirically keeping changes that work.

# Flow

Before starting work, backup the current script into the previous/ directory. Give it a filename based on the version, starting at 1 and increasing based on the existing files.

After you complete your change, make a git commit describing briefly what you changed. Put the accuracy at the start of the commit msg.
