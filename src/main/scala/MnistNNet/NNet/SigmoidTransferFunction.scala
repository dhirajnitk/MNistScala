package MnistNNet.NNet
import jeigen._
import jeigen.Shortcuts._;
trait SigmoidTransferFunction extends  TransferFunction {
   override def transfer(inducedLocalField : DenseMatrix) : DenseMatrix = {
    inducedLocalField.neg().exp().add(1).recpr()

  }
  override  def transferredSlope(inducedLocalField : DenseMatrix) : DenseMatrix = {
    val fixed = transfer(inducedLocalField);
    fixed.mul(fixed.neg().add(1))
  }
}
