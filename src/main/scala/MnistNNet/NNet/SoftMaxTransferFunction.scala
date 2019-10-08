package MnistNNet.NNet
import jeigen._
import jeigen.Shortcuts._
import Utility.OMP4JUtility
trait SoftMaxTransferFunction extends TransferFunction {
  override def transfer(inducedLocalField: DenseMatrix): DenseMatrix = {
    //ArrayXXf  inp = input.array() - input.maxCoeff();
    //return inp.cwiseProduct(inp.rowwise().sum().replicate(1, inp.cols()).inverse()).matrix();
     val inp = inducedLocalField.sub(inducedLocalField.maxOverCols().mmulj(ones(1,inducedLocalField.cols))).exp()
    inp.div(inp.sumOverCols().mmulj(ones(1,inp.cols)))


  }

  override def transferredSlope(inducedLocalField: DenseMatrix): DenseMatrix = {

    val dRows: Int = inducedLocalField.rows
    val dCols: Int = inducedLocalField.cols
    val S: DenseMatrix = transfer(inducedLocalField)
    var DS: DenseMatrix = zeros(dCols, dCols)
    val matDS: DenseMatrix = zeros(dRows, dCols * dCols)
    for (i <- 0 to dRows - 1) {
      val sMatrix = S.row(i).t().mmulj(ones(1, dCols))
      DS = sMatrix.mul(eye(dCols)).sub(sMatrix.mul(sMatrix.t()))
      //val table = DS.get2dValues().flatten
      // Row order = column order  for symmetric matrix
      val table = DS.getValues()
       //println(table(0),table(11),table(22))
      for (w <- 0 to dCols * dCols - 1) {
        matDS.set(i, w, table(w))
      }
    }
    matDS

   //OMP4JUtility.transferDerivative(transfer(inducedLocalField))
  }

  final def flatMult(left : DenseMatrix, right: DenseMatrix):DenseMatrix = {

    var  result:DenseMatrix  = new DenseMatrix(0,left.cols)
    for ( i <- 0 to left.rows -1 ){
      result = result.concatDown(left.row(i).mmul(new DenseMatrix(right.row(i).getValues().grouped(left.cols).toArray)))
    }
    result


    //OMP4JUtility.flatten(left, right)
  }
}