from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import TextOptions
from preprocessing import Preprocessing
from tokenization import Token

options = TextOptions()
opts = options.parse()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = Preprocessing().main()


    token = Token(opts)
    X_train_encoded, X_test_encoded= token.main(X_train, X_test)
    token.save(opts, X_train_encoded, X_test_encoded)

    trainer = Trainer(opts)
    trainer.train(X_train_encoded, X_test_encoded, y_train, y_test)
    trainer.save_model()
    trainer.evaluate(X_test_encoded, y_test)