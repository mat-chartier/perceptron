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
	
	public boolean classify(PerceptronResult perceptronResult, Integer[] data) throws PerceptronException {
		List<Integer> dataAsList = Arrays.asList(data);
		return classify(perceptronResult.getWeights(), dataAsList, perceptronResult.getWeightsFactors());
	}

	/**
	 * Returns true if above threshold (i.e scalar product is positive)
	 * @param weights
	 * @param data
	 * @param weightsFactors 
	 * @return
	 * @throws PerceptronException 
	 */
	public boolean classify(List<Integer> weights, List<Integer> data, int[] weightsFactors) throws PerceptronException {
		return conditionalScalarProduct(weights, data, weightsFactors) > 0;
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
			throw new PerceptronException("Training Data Vector is null or empty");
		}
		
		PerceptronThresholdData weights = new PerceptronThresholdData(data.size(), 0);
		
		// let's try and find the best weights for any combination of vector inputs
		List<IPerceptronTrainingData> validationData = validationDataSet.getTraingDataSet();
		List<int[]> allCombination = getAllCombination(weights.getTrainingVector().size());
		List<Integer> bestWeights = new ArrayList<>();
		long bestSuccessCount = 0;
		int[] bestWeightFactors = null;
		int combinationsCount = allCombination.size() - 1;
		System.out.println("Nb combinations = " + combinationsCount);
		int currentCombination = 0;
		int currentPercent = 0;
		for (int i = 0; i < 100; i++) {
			System.out.print(".");
		}
		System.out.println();
		for (int[] currentWeightsFactors : allCombination) {
			currentCombination++;
			if (isZeroVector(currentWeightsFactors)) {
				continue;
			}
			
			int percent = currentCombination * 100 / combinationsCount;
			if (percent > currentPercent) {
				for (int j = 0; j < (percent - currentPercent) && j + currentPercent < 100; j++) {
					System.out.print("*");
				}
				currentPercent = percent;
			}
			
			
			List<Integer> currentWeights = new ArrayList<>();
			for (int count = 0 ; count < maxLearningLoops; count++) {
				if (dataSetSize != trainingDataSet.getTraingDataSet().size()) {
					throw new PerceptronException("Perceptron logic error, data set size is not constant !");
				}
				
				if (currentWeights.isEmpty()) {
					currentWeights = learn(trainingDataSet, weights.getTrainingVector(), currentWeightsFactors);
					
					long successCount = check(validationData, weights.getTrainingVector(), currentWeightsFactors);
					if (successCount > bestSuccessCount) {
						bestWeightFactors = currentWeightsFactors;
						bestWeights = currentWeights;
						bestSuccessCount = successCount; 
					}
				} else {
					List<Integer> newWeights = learn(trainingDataSet, currentWeights, currentWeightsFactors);
					if (sameVector(bestWeights, newWeights)) {
						break;
					} else {
						currentWeights = newWeights;
						long currentSuccessCount = check(validationData, currentWeights, currentWeightsFactors);
						
						if (currentSuccessCount > bestSuccessCount) {
							bestSuccessCount = currentSuccessCount;
							bestWeights = currentWeights;
							bestWeightFactors = currentWeightsFactors.clone();
						}
					}
				}
			}
		}
		
		System.out.println();
		PerceptronResult ret = new PerceptronResult((int)(bestSuccessCount * 100 / validationData.size()), bestWeights, bestWeightFactors);
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

	private long check(List<IPerceptronTrainingData> traingDataSet, List<Integer> weights, int[] weightsFactors) throws PerceptronException {
		long success = 0;
		for (IPerceptronTrainingData perceptronTrainingData : traingDataSet) {
			long scalarProduct = conditionalScalarProduct(perceptronTrainingData.getTrainingVector(), weights, weightsFactors);
			if (perceptronTrainingData.isAboveThreshold()) {
				success += scalarProduct > 0 ? 1: 0;
			} else {
				success += scalarProduct <= 0 ? 1: 0;
			}
		}
		return success;
	}

	private List<Integer> learn(IPerceptronDataSet trainingDataSet, List<Integer> weights, int[] weightsFactors) throws PerceptronException {
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
			long scalarProduct = conditionalScalarProduct(trainingData, weights, weightsFactors);
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

	private long conditionalScalarProduct(List<Integer> data, List<Integer> weights, int[] weightsFactors) throws PerceptronException {
		if (data == null) {
			throw new PerceptronException("Data list is null");
		}
		if (weights == null) {
			throw new PerceptronException("Weights list is null");
		}
		int dataLength = data.size();
		int weightsLength = weights.size();
		if (dataLength != weightsLength) {
			throw new PerceptronException("Vectors must have same size to compute scalar product : data size = " + dataLength + ", weights size = " + weightsLength);
		}
		
		long ret = 0l;
		
		for (int i = 0; i < dataLength; i++) {
			ret += data.get(i) * weights.get(i) * weightsFactors[i];
		}
		
		return ret ;
	}
	
	private boolean isZeroVector(int[] vector) {
		for (int value : vector) {
			if (value != 0) {
				return false;
			}
		}
		return true;
	}
	
	private List<int[]> getAllCombination(int size) throws PerceptronException {
		if (size > 15) {
			throw new PerceptronException("Maximum number of parameters on vector is 15");
		}
	    int numRows = (int) Math.pow(2, size);
	    boolean[][] bools = new boolean[numRows][size];
	    List<int[]> res = new ArrayList<int[]>();
	    for(int i = 0;i<bools.length;i++)
	    {
	        for(int j = 0; j < bools[i].length; j++)
	        {
	            int val = bools.length * j + i;
	            int ret = (1 & (val >>> j));
	            bools[i][j] = ret != 0;
	        }
	    }
	    
	    for (int i = 0; i < bools.length; i++) {
	    	int[] row = new int[size];
	    	for (int j = 0; j < size; j++) {
	    		row[j] = bools[i][j] ? 1 : 0;
	    	}
	    	res.add(row);
	    }
	    return res;
	}
}
