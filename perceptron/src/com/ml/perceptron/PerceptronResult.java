package com.ml.perceptron;

import java.util.ArrayList;
import java.util.List;

public class PerceptronResult {

	private int accuracy;
	private List<Integer> weights;

	public PerceptronResult(int accuracy, List<Integer> weights) {
		super();
		this.accuracy = accuracy;
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

	@Override
	public String toString() {
		return "PerceptronResult [accuracy=" + accuracy + ", weights=" + weights + "]";
	}

}
