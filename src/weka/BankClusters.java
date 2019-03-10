package weka;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class BankClusters {
    public static void main(String[] args) throws Exception {
        //
        DataSource source = new DataSource("C:\\Users\\pomia\\Desktop\\MachineLearning\\data\\bank-data\\bank-data.arff");

        Instances data = source.getDataSet();

        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(8);
        kmeans.buildClusterer(data);
        System.out.println(kmeans);
    }
}
