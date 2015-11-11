package moa.classifiers.functions;

import weka.core.Instance;
import weka.core.Utils;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

public class CountClasses extends AbstractClassifier {

	private static final long serialVersionUID = -2340429651926966699L;
	
	private double[] m_classDist = null;

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		
		if(m_classDist == null) {
			m_classDist = new double[ inst.numClasses() ];
		}
		m_classDist[ (int) inst.classValue() ] += 1;
		// now print the class dist
		double[] norm = new double[m_classDist.length];
		for(int x = 0; x < norm.length; x++) norm[x] = m_classDist[x];
		Utils.normalize(norm);
		for(int x = 0; x < norm.length; x++) {
			System.out.print(norm[x]);
			if(x != norm.length-1) {
				System.out.print(",");
			}
		}
		System.out.println("");
		
		double[] preds = new double[ inst.numClasses() ];
		preds[0] = 1;
		return preds;
	}

	@Override
	public void resetLearningImpl() { }

	@Override
	public void trainOnInstanceImpl(Instance inst) { }

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) { }
	

}
