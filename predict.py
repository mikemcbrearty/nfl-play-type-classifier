import numpy as np
from tensorflow import keras

from data_loader import load_data

# Script to aid analysis of Keras model results 


model = keras.models.load_model("play_data_dense.keras")
play_type = ["pass", " run", "punt", "field_goal"]
down_list = ["", "1st", "2nd", "3rd", "4th"]

def format_input(input):
    down = down_list[int(input[0] * 4)]
    ydstogo = input[1] * 10
    ydstogo_100 = input[2] * 100
    if input[3] > 0:
        score_diff = f"  up {input[3] * 60:2.0f}"
    elif input[3] < 0:
        score_diff = f"down {abs(input[3]) * 60:2.0f}"
    else:
        score_diff = "   tied"
    seconds = int(input[5] * 1800)
    if input[4] == 0.0 and seconds > 900:
        quarter = "Q1"
    elif input[4] == 0.0:
        quarter = "Q2"
    elif seconds > 900:
        quarter = "Q3"
    else:
        quarter = "Q4"
    seconds = seconds % 900 if seconds > 900 else seconds
    time = f"{seconds // 60:2}:{seconds % 60:02}"

    input_clause = f", {down} & {ydstogo:2.0f}, at {ydstogo_100:2.0f}, {score_diff}, {quarter} {time}"
    return input_clause


def make_prediction(
    inputs,
    targets=None,
    filter_to_target=None,
    filter_to_down=None,
    filter_to_result=None,
    print_each_prediction=True,
):
    predictions = model.predict(inputs)
    num_predictions = 0
    num_correct_predictions = 0
    for i, prediction in enumerate(predictions):
        if targets is not None and filter_to_target is not None:
            if targets[i][filter_to_target] != 1:
                continue
        if filter_to_down is not None and int(inputs[i][0] * 4) != filter_to_down:
            continue

        j_1 = np.argmax(prediction)
        copy_prediction = np.copy(prediction)
        copy_prediction[j_1] = 0
        j_2 = np.argmax(copy_prediction)

        input_clause = format_input(inputs[i])

        if targets is None:
            target_clause = ""
        else:
            k = np.argmax(targets[i])

            if filter_to_result is not None and ((j_1 == k) != filter_to_result):
                continue

            target_clause = f", target: {play_type[k]} ({' True' if j_1 == k else 'False'})"
            num_predictions += 1
            if j_1 == k:
                num_correct_predictions += 1
        if print_each_prediction:
            print(f"prediction: {play_type[j_1]} ({prediction[j_1]*100:.1f}%) {play_type[j_2]} ({prediction[j_2]*100:>4.1f}%){target_clause}{input_clause}")
    if targets is not None:
        print(f"number of prediction: {num_predictions}, number correct: {num_correct_predictions} ({num_correct_predictions/num_predictions*100:.1f}%)")

# Goal line situation from Super Bowl 49
# make_prediction(np.array([[2/4., 2/10., 2/100., -4/60., 1., 24/1800., 1/3., 2/3.]]))

(train_data, val_data, test_data), (train_targets, val_targets, test_targets) = load_data()

num_samples = 200
make_prediction(
    val_data[:num_samples],
    targets=val_targets[:num_samples],
    filter_to_target=1,
    filter_to_down=None,
    filter_to_result=False,
)
