package MnistNNet.NNet
import jeigen._
trait TransferFunction extends AnyRef {
  def transfer(inducedLocalField : DenseMatrix) : DenseMatrix
  def transferredSlope(inducedLocalField : DenseMatrix) : DenseMatrix
}
