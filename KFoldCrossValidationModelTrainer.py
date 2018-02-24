from DataGenerator import RandomizeKFoldDataGeneratorPairGenerator, ApplicationDirPathGetter
from VGG16AutoEncoderGenerator import VGG16AutoEncoderGenerator
import numpy as np
import os


def save_model_as_json(model, target_file_path):
    model_in_json = model.to_json()
    with open(target_file_path, "w") as file:
        file.write(model_in_json)


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model_and_weight_storing_dir = "model_and_weight"

    base_data_source_dir = 'data_source'

    randomize_k_fold_data_generator_pair_generator = \
        RandomizeKFoldDataGeneratorPairGenerator().get_training_and_validation_data_generator_pair(
            num_of_folds=10,
            data_source_dir=os.path.join(ApplicationDirPathGetter().execute(), "data_source", "training"),
            batch_size_of_training_data_set=64,
            batch_size_of_validation_data_set=100
        )

    cross_validation_loss = list()
    cross_validation_accuracy = list()

    for training_data_generator, validation_data_generator in randomize_k_fold_data_generator_pair_generator:
        vgg16_auto_encoder_generator = VGG16AutoEncoderGenerator(
            input_shape=input_shape,
            weight_file=os.path.join(model_and_weight_storing_dir, 'vgg16_weights_notop.h5'),
            number_of_layers_to_freeze_from_input_layer=10
        )

        vgg16_auto_encoder_generator.auto_encoder.compile(
            loss='binary_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy']
        )

        vgg16_auto_encoder_generator.auto_encoder.fit_generator(
            generator=training_data_generator.infinitely_generate_batch_of_data_pair_tuple(),
            steps_per_epoch=300, epochs=150, verbose=2)

        scores = vgg16_auto_encoder_generator.auto_encoder.evaluate_generator(
            generator=validation_data_generator.infinitely_generate_batch_of_data_pair_tuple(),
            steps=20
        )
        print(vgg16_auto_encoder_generator.auto_encoder.metrics_names)
        print(scores)
        cross_validation_loss.append(scores[0])
        cross_validation_accuracy.append(scores[1])

        if scores[0] <= min(cross_validation_loss) and scores[1] >= max(cross_validation_accuracy):
            print("save encoder as 'encoder_model.json' and 'encoder_weight.h5'")
            save_model_as_json(
                vgg16_auto_encoder_generator.encoder, os.path.join(model_and_weight_storing_dir, "encoder_model.json")
            )
            vgg16_auto_encoder_generator.encoder.save_weights(
                os.path.join(model_and_weight_storing_dir, 'encoder_weight.h5')
            )

            print("save auto encoder as 'auto_encoder_model.json' and 'auto_encoder_weight.h5")
            save_model_as_json(
                vgg16_auto_encoder_generator.auto_encoder,
                os.path.join(model_and_weight_storing_dir, "auto_encoder_model.json")
            )
            vgg16_auto_encoder_generator.auto_encoder.save_weights(
                os.path.join(model_and_weight_storing_dir, 'auto_encoder_weight.h5')
            )

    print("==========loss==========")
    print(cross_validation_loss)
    print("{}% (+- {}%)".format(np.mean(cross_validation_loss) * 100, np.std(cross_validation_loss) * 100))

    print("==========accuracy==========")
    print(cross_validation_accuracy)
    print("{}% (+- {}%)".format(np.mean(cross_validation_accuracy) * 100, np.std(cross_validation_accuracy) * 100))
