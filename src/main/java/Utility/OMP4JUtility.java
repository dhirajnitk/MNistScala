package Utility;
import static jeigen.Shortcuts.*;
import jeigen.*;
public class OMP4JUtility{

public static DenseMatrix transferDerivative(DenseMatrix transferField) {
	int dRows = transferField.rows;
    int dCols = transferField.cols;
    //DenseMatrix DS = zeros(dCols, dCols);
    DenseMatrix matDS = zeros(dRows, dCols * dCols);
    double [] table = new double[dCols * dCols];
    DenseMatrix sMatrix;
    int w;
	// omp parallel for private(sMatrix,table,w)
    for (int i = 0; i<dRows; i++) {
      sMatrix = transferField.row(i).t().mmulj(ones(1, dCols));
      table = sMatrix.mul(eye(dCols)).sub(sMatrix.mul(sMatrix.t())).getValues();
      //val table = DS.get2dValues().flatten
      // Row order = column order  for symmetric matrix
       //println(table(0),table(11),table(22))
      for (w = 0; w < dCols * dCols ; w++) {
        matDS.set(i, w, table[w]);
      }
    }
    return matDS;
} 
private static double[][] grouped(double[] data,int cols) {
	final int rows = data.length/cols;
	double[][] table = new double[rows][cols];
    int col;
	// omp parallel for private(col)
	for(int row = 0; row < rows; row++)
	{  
		for(col = 0; col < cols; col++)
		{
			table[row][col] = data[row * cols + col];
		}
	}
	return table;
} 
 public static DenseMatrix flatten(DenseMatrix left, DenseMatrix right) {
     double[] result  = new double[left.rows * left.cols];
     double [] temp =  new double[left.cols * left.cols];
	// omp parallel for private(temp)
    for ( int i =0; i <left.rows;i++){
        temp = (left.row(i).mmul(new DenseMatrix(grouped(right.row(i).getValues(),left.cols)))).getValues();
	    System.arraycopy(temp, 0, result, i* temp.length, temp.length);
    }
    return new DenseMatrix(left.cols, left.rows, result).t();
	 
 }

}  