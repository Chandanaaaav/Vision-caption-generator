import cv2


def open_camera():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for i in range(5):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend) if backend -ne $null else cv2.VideoCapture(i)
            except Exception {
                continue
            }

            if (cap.isOpened()) {
                return ,$cap,$i,$backend
            }

            $cap.release()

    return ,$null,$null,$null


$cap,$camera_index,$backend = open_camera()

if ($cap -ne $null) {
    $backend_name = if ($backend -eq [OpenCvSharp.Cv2]::CAP_DSHOW) { "DirectShow" } else { "Default" }
    Write-Host "✅ Camera opened at index $camera_index using $backend_name backend."

    while ($true) {
        $ret, $frame = $cap.read()
        if (-not $ret) {
            Write-Host "❌ Camera opened but cannot grab frame."
            break
        }

        cv2.imshow("Camera $camera_index", $frame)
        $key = [System.Console]::ReadKey($true).KeyChar
        if ($key -eq 'q') {
            break
        }
    }

    $cap.release()
    cv2.destroyAllWindows()
} else {
    Write-Host "❌ No working camera found."
}
