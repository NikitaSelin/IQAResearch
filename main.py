from data import create_data_lists, view_random_image


if __name__ == "__main__":
    create_data_lists(["DIV2K_train_HR"], ["DIV2K_valid_HR"], 257, "jsons")
    view_random_image("jsons/train_images.json")
