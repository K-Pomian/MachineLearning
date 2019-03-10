package weka;

import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DiabetesClassification {
    public static void main(String[] args) throws Exception {
        DataSource trainDataSource = new DataSource("C:\\Users\\pomia\\Desktop\\MachineLearning\\data\\Diabetes\\diabetes.arff");

        Instances trainDataset = trainDataSource.getDataSet();
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

        SMO smo = new SMO();
        System.out.println(smo.getCapabilities().toString());
        smo.buildClassifier(trainDataset);
        System.out.println(smo.toString());

        DataSource testDataSource = new DataSource("C:\\Users\\pomia\\Desktop\\MachineLearning\\data\\Diabetes\\diabetes_test.arff");

        Instances testDataset = testDataSource.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        System.out.println("===================");
        System.out.println("Actual Class, SMO Predicted");

        Instance newInst;
        double actualValue;
        double predSMO;
        for (int i =  0; i < testDataset.numInstances(); i++) {
            actualValue = testDataset.instance(i).classValue();
            newInst = testDataset.instance(i);
            predSMO = smo.classifyInstance(newInst);

            System.out.println(actualValue + ", " + predSMO);
        }
    }

}
