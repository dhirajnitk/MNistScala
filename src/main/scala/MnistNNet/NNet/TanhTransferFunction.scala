package MnistNNet.NNet
import jeigen._
trait  TanhTransferFunction extends  TransferFunction {
  /*
   final def realTransfer(inducedLocalField : DenseMatrix) : DenseMatrix = {
     inducedLocalField.mul(2).neg().exp().add(1).recpr().mul(2).sub(1)
   }
  final def realTransferSlope(inducedLocalField:DenseMatrix):DenseMatrix = {
    val tempTransfer: DenseMatrix = transfer(inducedLocalField)
    tempTransfer.pow(2).neg().add(1)
  }
   override  def transfer(inducedLocalField : DenseMatrix) : DenseMatrix = {
    realTransfer(inducedLocalField.mul(0.6667)).mul(1.7159)
  }
  override  def transferredSlope(inducedLocalField : DenseMatrix) : DenseMatrix = {
    realTransferSlope(inducedLocalField.mul(0.6667)).mul(1.14393)
  }
  */

  override  def transfer(inducedLocalField : DenseMatrix) : DenseMatrix = {
    inducedLocalField.mul(2).neg().exp().add(1).recpr().mul(2).sub(1)
  }
  override  def transferredSlope(inducedLocalField : DenseMatrix) : DenseMatrix = {
    val tempTransfer: DenseMatrix = transfer(inducedLocalField)
    tempTransfer.pow(2).neg().add(1)
  }
}
