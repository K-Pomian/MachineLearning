package weka;

import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MarketBasketAssociation {
    public static void main(String[] args) throws Exception {
        DataSource dataSource = new DataSource("C:\\Users\\pomia\\Desktop\\MachineLearning\\data\\market-basket\\marketbasket.arff");

        Instances data = dataSource.getDataSet();

        Apriori model = new Apriori();
        String[] options = {"-N", "20", "-T", "0", "-C", "0.9", "-D", "0.01", "-U", "1", "-M", "0.5", "-S", "-1.0", "-c", "-1"};

        model.setOptions(options);
        model.buildAssociations(data);
        System.out.println(model);

        FPGrowth fpGrowth = new FPGrowth();
        fpGrowth.setNumRulesToFind(20);
        fpGrowth.setLowerBoundMinSupport(0.01);
        fpGrowth.setMinMetric(0.1);
        fpGrowth.buildAssociations(data);

        System.out.println(fpGrowth);
    }
}
