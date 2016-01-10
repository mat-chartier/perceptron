package com.ml.perceptron;

import java.util.ArrayList;
import java.util.List;

public class PerceptronEngineTest {

	
	public static void main(String[] args) throws PerceptronException {
		PerceptronEngine engine = new PerceptronEngine();
		IPerceptronDataSet trainingDataSet = new IPerceptronDataSet() {
			
			@Override
			public List<IPerceptronTrainingData> getTraingDataSet() {
				List<IPerceptronTrainingData> ret = new ArrayList<IPerceptronTrainingData>();
				ret.add(new t(1, false));
				ret.add(new t(2, false));
				ret.add(new t(3, false));
				ret.add(new t(4, false));
				ret.add(new t(6, true));
				ret.add(new t(7, true));
				return ret;
			}
		};
		PerceptronResult learn = engine.learn(trainingDataSet, trainingDataSet);
		System.out.println(learn);
		
		System.out.println("Should be true : " + engine.classify(learn, new Integer[] {10, 8}));
		System.out.println("Should be false : " + engine.classify(learn, new Integer[] {10, 0}));
		
	}
	
	static class t implements IPerceptronTrainingData {

		private Integer value;
		private boolean above;

		public t(Integer value, boolean above) {
			super();
			this.value = value;
			this.above = above;
		}

		@Override
		public List<Integer> getTrainingVector() {
			List<Integer> data = new ArrayList<Integer>();
			data.add(10);
			data.add(value);
			return data;
		}

		@Override
		public boolean isAboveThreshold() {
			return above;
		}
		
	}
}
