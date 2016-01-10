package com.ml.perceptron;

import java.util.List;

public interface IPerceptronTrainingData {

	public List<Integer> getTrainingVector();

	public boolean isAboveThreshold();
}
