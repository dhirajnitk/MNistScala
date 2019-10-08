package MnistNNet.NNet
import jeigen._
import jeigen.Shortcuts._
trait ReLUTransferFunction extends TransferFunction {
  override  def transfer(inducedLocalField : DenseMatrix) : DenseMatrix = {
    //Default Non leaky
    inducedLocalField.max(0)
    //Leaky mode
    //inducedLocalField.max(0.01)
  }
  override  def transferredSlope(inducedLocalField : DenseMatrix) : DenseMatrix = {
      for {i <- 0 to inducedLocalField.rows - 1
        j <- 0 to inducedLocalField.cols - 1
      }
      {
        if(inducedLocalField.get(i,j)<0)
          inducedLocalField.set(i,j,0)
        else
          inducedLocalField.set(i,j,1)
      }

    inducedLocalField
    }
}
