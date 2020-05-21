from sklearn.model_selection import train_test_split

class Preprocess:
    def split(self, input, output, file):
        x = file[input]
        y = file[output]
        x_changed = self.__to_float(x)
        return train_test_split(x_changed, y, test_size=0.2, random_state=5)

    def __to_float(self, input):
        return input.astype(float)