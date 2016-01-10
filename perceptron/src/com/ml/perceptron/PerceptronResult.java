package com.ml.perceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PerceptronResult {

	private int accuracy;
	private List<Integer> weights;
	private int[] weightsFactors;

	public PerceptronResult(int accuracy, List<Integer> weights, int[] weightsFactors) {
		super();
		this.accuracy = accuracy;
		this.weightsFactors = weightsFactors.clone();
		this.weights = new ArrayList<>();
		// real cloning
		for (Integer weight : weights) {
			this.weights.add(weight.intValue());
		}
	}

	public int getAccuracy() {
		return accuracy;
	}

	public List<Integer> getWeights() {
		return weights;
	}

	public int[] getWeightsFactors() {
		return weightsFactors;
	}

	@Override
	public String toString() {
		return "PerceptronResult [accuracy=" + accuracy + ", weights=" + weights + ", weightsFactors="
				+ Arrays.toString(weightsFactors) + "]";
	}

}
