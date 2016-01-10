package com.ml.perceptron;

import java.util.ArrayList;
import java.util.List;

public class PerceptronThresholdData implements IPerceptronTrainingData {

	private List<Integer> trainingData;
	
	

	public PerceptronThresholdData(int dataCount, int defaultValue) {
		trainingData = new ArrayList<Integer>();
		
		for (int i = 0; i < dataCount; i++) {
			trainingData.add(defaultValue);
		}
	}



	@Override
	public List<Integer> getTrainingVector() {
		return trainingData;
	}

	@Override
	public boolean isAboveThreshold() {
		return true;
	}

}
