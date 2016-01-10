package com.ml.perceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class PerceptronEngine {

	private int maxLearningLoops = 1000;

	public PerceptronEngine() {}
	
	public PerceptronEngine(int maxLearningLoops) {
		this.maxLearningLoops = maxLearningLoops;
	}
	
	public boolean classify(List<Integer> weights, Integer[] data) throws PerceptronException {
		List<Integer> dataAsList = Arrays.asList(data);
		return classify(weights, dataAsList);
	}

	/**
	 * Returns true if above threshold (i.e scalar product is positive)
	 * @param weights
	 * @param data
	 * @return
	 * @throws PerceptronException 
	 */
	public boolean classify(List<Integer> weights, List<Integer> data) throws PerceptronException {
		return scalarProduct(weights, data) > 0;
	}
	
	public PerceptronResult learn(IPerceptronDataSet trainingDataSet, IPerceptronDataSet validationDataSet) throws PerceptronException {
		if (trainingDataSet == null) {
			throw new PerceptronException("Training Data Set is null");
		}
		
		List<IPerceptronTrainingData> dataSet = trainingDataSet.getTraingDataSet();
		if (dataSet == null || dataSet.isEmpty()) {
			throw new PerceptronException("Training Data List is null or empty");
		}
		int dataSetSize = dataSet.size();
		List<Integer> data = dataSet.get(0).getTrainingVector();
		if (data == null || data.isEmpty()) {
			throw new PerceptronException("Training Data List is null or empty");
		}
		
		PerceptronThresholdData weights = new PerceptronThresholdData(data.size(), 0);
		
		// let's try and find the best weights
		List<Integer> bestWeights = learn(trainingDataSet, weights.getTrainingVector());
		List<IPerceptronTrainingData> validationData = validationDataSet.getTraingDataSet();

		long nbSuccess = check(validationData, bestWeights);
		List<Integer> currentWeights = learn(trainingDataSet, bestWeights);

		for (int count = 0 ; count < maxLearningLoops; count++) {
			if (dataSetSize != trainingDataSet.getTraingDataSet().size()) {
				throw new PerceptronException("Perceptron logic error, data set size is not constant !");
			}
			if (sameVector(bestWeights, currentWeights)) {
				break;
			}
			long currentNbSuccess = check(validationData, currentWeights);
			if (currentNbSuccess > nbSuccess) {
				nbSuccess = currentNbSuccess;
				bestWeights = currentWeights;
			}
			currentWeights = learn(trainingDataSet, currentWeights);
		}
		
		PerceptronResult ret = new PerceptronResult((int)(nbSuccess * 100 / validationData.size()), bestWeights);
		return ret;
	}

	private boolean sameVector(List<Integer> vector1, List<Integer> vector2) {
		Iterator<Integer> iterator1 = vector1.iterator();
		Iterator<Integer> iterator2 = vector2.iterator();
		while (iterator1.hasNext()) {
			Integer value1 = (Integer) iterator1.next();
			Integer value2 = (Integer) iterator2.next();
			if (value1.intValue() != value2.intValue()) {
				return false;
			}
		}
		return true;
	}

	private long check(List<IPerceptronTrainingData> traingDataSet, List<Integer> weights) throws PerceptronException {
		long success = 0;
		for (IPerceptronTrainingData perceptronTrainingData : traingDataSet) {
			long scalarProduct = scalarProduct(perceptronTrainingData.getTrainingVector(), weights);
			if (perceptronTrainingData.isAboveThreshold()) {
				success += scalarProduct > 0 ? 1: 0;
			} else {
				success += scalarProduct <= 0 ? 1: 0;
			}
		}
		return success;
	}

	private List<Integer> learn(IPerceptronDataSet trainingDataSet, List<Integer> weights) throws PerceptronException {
		if (trainingDataSet == null) {
			throw new PerceptronException("Training Data Set is null");
		}
		
		List<IPerceptronTrainingData> dataSet = trainingDataSet.getTraingDataSet();
		if (dataSet == null || dataSet.isEmpty()) {
			throw new PerceptronException("Training Data List is null or empty");
		}
		// Add threshold
		List<Integer> data = dataSet.get(0).getTrainingVector();
		if (data == null || data.isEmpty()) {
			throw new PerceptronException("Training Data List is null or empty");
		}
		
		dataSet.add(0, new PerceptronThresholdData(data.size(), 1));
		
		// Loop over the training data set
		for (IPerceptronTrainingData perceptronTrainingData : dataSet) {
			List<Integer> trainingData = perceptronTrainingData.getTrainingVector();
			long scalarProduct = scalarProduct(trainingData, weights);
			boolean shoudlBePositive = perceptronTrainingData.isAboveThreshold();
			
			if (scalarProduct < 0) {
				if (shoudlBePositive) {
					// Add training data to weights and stop
					weights = add(trainingData, weights);
					break;
				}
			} else {
				if (!shoudlBePositive) {
					// Deduct training data from weights and stop
					weights = deduct(trainingData, weights);
					break;
				}
			}
		}
		
		// Remove threshold data
		dataSet.remove(0);
		
		return weights;
	}
	
	private List<Integer> deduct(List<Integer> trainingData, List<Integer> weights) {
		List<Integer> ret = new ArrayList<>();
		for (int i = 0 ; i < trainingData.size(); i++) {
			ret.add(weights.get(i) - trainingData.get(i));
		}
		return ret;
	}

	private List<Integer> add(List<Integer> trainingData, List<Integer> weights) {
		List<Integer> ret = new ArrayList<>();
		for (int i = 0 ; i < trainingData.size(); i++) {
			ret.add(weights.get(i) + trainingData.get(i));
		}
		return ret;
	}

	private long scalarProduct(List<Integer> data, List<Integer> weights) throws PerceptronException {
		if (data == null) {
			throw new PerceptronException("Data list is null");
		}
		if (weights == null) {
			throw new PerceptronException("Weights list is null");
		}
		int dataLength = data.size();
		int weightsLength = weights.size();
		if (dataLength != weightsLength) {
			throw new PerceptronException("Vectors must have same size to compute scalara product : data size = " + dataLength + ", weights siz = " + weightsLength);
		}
		
		long ret = 0l;
		
		for (int i = 0; i < dataLength; i++) {
			ret += data.get(i) * weights.get(i);
		}
		
		return ret ;
	}
}
