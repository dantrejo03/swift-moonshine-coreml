import CoreML
import CoreVideo

// MLMultiArray is Core ML's tensor: a shape, a number type and a block of memory. Its strides say how many elements
// apart neighbours on each axis are. Arrays made with makeZeroed are row-major with no padding, so they can be indexed
// directly. Arrays Core ML returns may be padded (it picks their layout, for example to suit the Neural Engine), and
// so may surface-backed ones, so every read or write of those goes through the strides they report.

extension MLMultiArray {
    var integerShape: [Int] { shape.map(\.intValue) }
    var integerStrides: [Int] { strides.map(\.intValue) }

    /// A new array of zeros, row-major with no padding. Checked, since code that fills it by index relies on that.
    static func makeZeroed(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
        let expectedStrides = rowMajorStrides(for: shape)
        guard array.integerStrides == expectedStrides else {
            throw Moonshine.Error(
                "a new \(shape) array has strides \(array.integerStrides), not row-major \(expectedStrides)")
        }
        array.withUnsafeMutableBytes { bytes, _ in
            _ = bytes.initializeMemory(as: UInt8.self, repeating: 0)
        }
        return array
    }

    /// A new float16 array of zeros backed by an IOSurface: memory the CPU and the Neural Engine share, so Core ML can
    /// pass it to a model without copying it on every prediction. Underneath is a pixel buffer as wide as the shape's
    /// last axis and as tall as the product of the others. Its rows may be padded, so write it through its strides.
    static func makeZeroedSurfaceBackedFloat16(shape: [Int]) throws -> MLMultiArray {
        guard let width = shape.last else {
            throw Moonshine.Error("a surface-backed array needs at least one axis")
        }
        let height = shape.dropLast().reduce(1, *)
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent16Half, attributes, &pixelBuffer)
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw Moonshine.Error(
                "couldn't create a \(width) × \(height) float16 pixel buffer for a \(shape) array (CVReturn \(status))")
        }
        let array = MLMultiArray(pixelBuffer: pixelBuffer, shape: shape.map { NSNumber(value: $0) })
        array.withUnsafeMutableBytes { bytes, _ in
            _ = bytes.initializeMemory(as: UInt8.self, repeating: 0)
        }
        return array
    }

    /// Each axis's stride when the array is row-major with no padding: the last axis is 1, and each earlier axis is the
    /// product of the sizes after it.
    static func rowMajorStrides(for shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        for axis in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        }
        return strides
    }
}

extension MLMultiArrayDataType {
    /// The number type's name, for error messages.
    var name: String {
        switch self {
        case .float32: "float32"
        case .float16: "float16"
        case .int32: "int32"
        case .double: "float64"
        default: "data type \(rawValue)"
        }
    }
}
