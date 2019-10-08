package MnistNNet.NNet
import jeigen._
import jeigen.Shortcuts._
trait SoftPlusTransferFunction extends TransferFunction {
  override def transfer(inducedLocalField: DenseMatrix): DenseMatrix = {
    inducedLocalField.exp().add(1).log()
  }

  override def transferredSlope(inducedLocalField: DenseMatrix): DenseMatrix = {
    inducedLocalField.neg().exp().add(1).recpr()
  }
}