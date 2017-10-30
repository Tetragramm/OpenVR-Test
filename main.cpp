//===================== Copyright (c) Valve Corporation. All Rights Reserved. ======================
//==================================================================================================

#include <vector>
#include <iostream>
#include <ctime>
#include "opencv2\core.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\calib3d.hpp"
#include "opencv2\video.hpp"
#include "opencv2\aruco\charuco.hpp"
#include <openvr.h>


using namespace cv;

static bool saveCameraParams(const String &filename, Size imageSize, float aspectRatio, int flags,
    const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        return false;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;

    return true;
}

int main(int argc, char *argv[])
{
    vr::CameraVideoStreamFrameHeader_t m_CurrentFrameHeader;
    
    vr::IVRSystem					*m_pVRSystem;
    vr::IVRTrackedCamera			*m_pVRTrackedCamera;

    vr::TrackedCameraHandle_t	m_hTrackedCamera;

    uint32_t				m_nLastFrameSequence = 0;

    uint32_t				m_nCameraFrameWidth;
    uint32_t				m_nCameraFrameHeight;
    uint32_t				m_nCameraFrameBufferSize;
    vr::HmdVector2_t        m_nFocalLength, m_nCenter;
    cv::Mat                 image, validMask;
    std::vector<cv::Mat>    imgBuffer, poseBuffer;

    std::cout<<"\nStarting OpenVR...\n";
    vr::EVRInitError eError = vr::VRInitError_None;
    m_pVRSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);
    if (eError != vr::VRInitError_None)
    {
        m_pVRSystem = nullptr;
        std::cout << "Unable to init VR runtime:" << vr::VR_GetVRInitErrorAsSymbol(eError) << "\n";
        return 1;
    }
    else
    {
        char systemName[1024];
        char serialNumber[1024];
        m_pVRSystem->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String, systemName, sizeof(systemName));
        m_pVRSystem->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String, serialNumber, sizeof(serialNumber));
    }

    m_pVRTrackedCamera = vr::VRTrackedCamera();
    if (!m_pVRTrackedCamera)
    {
       std::cout<<"Unable to get Tracked Camera interface.\n";
        return 1;
    }

    bool bHasCamera = false;
    vr::EVRTrackedCameraError nCameraError = m_pVRTrackedCamera->HasCamera(vr::k_unTrackedDeviceIndex_Hmd, &bHasCamera);
    if (nCameraError != vr::VRTrackedCameraError_None || !bHasCamera)
    {
        std::cout<<"No Tracked Camera Available! "<<m_pVRTrackedCamera->GetCameraErrorNameFromEnum(nCameraError)<<"\n";
        return 1;
    }

    // Accessing the FW description is just a further check to ensure camera communication is valid as expected.
    vr::ETrackedPropertyError propertyError;
    char buffer[128];
    m_pVRSystem->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_CameraFirmwareDescription_String, buffer, sizeof(buffer), &propertyError);
    if (propertyError != vr::TrackedProp_Success)
    {
        std::cout<<"Failed to get tracked camera firmware description!\n";
        return 1;
    }
    
    nCameraError = m_pVRTrackedCamera->GetCameraFrameSize(vr::k_unTrackedDeviceIndex_Hmd, vr::VRTrackedCameraFrameType_Undistorted, &m_nCameraFrameWidth, &m_nCameraFrameHeight, &m_nCameraFrameBufferSize);
    if (nCameraError != vr::VRTrackedCameraError_None)
    {
        std::cout<<"GetCameraFrameBounds() Failed!\n";
        std::cout << m_pVRTrackedCamera->GetCameraErrorNameFromEnum(nCameraError) << "\n";
        return 1;
    }

    nCameraError = m_pVRTrackedCamera->GetCameraIntrinisics(vr::k_unTrackedDeviceIndex_Hmd, vr::VRTrackedCameraFrameType_Undistorted, &m_nFocalLength, &m_nCenter);
    if (nCameraError != vr::VRTrackedCameraError_None)
    {
        std::cout << "GetCameraIntrinisics() Failed!\n";
        std::cout << m_pVRTrackedCamera->GetCameraErrorNameFromEnum(nCameraError) << "\n";
        return 1;
    }

    if (m_nCameraFrameBufferSize / (m_nCameraFrameHeight*m_nCameraFrameWidth) == 4)
    {
        image.create(m_nCameraFrameHeight, m_nCameraFrameWidth, CV_8UC4);
    }
    else if (m_nCameraFrameBufferSize / (m_nCameraFrameHeight*m_nCameraFrameWidth) == 3)
    {
        image.create(m_nCameraFrameHeight, m_nCameraFrameWidth, CV_8UC3);
    }
    else
    {
        std::cout << "Error in Frame Size\n";
    }

    m_pVRTrackedCamera->AcquireVideoStreamingService(vr::k_unTrackedDeviceIndex_Hmd, &m_hTrackedCamera);
    if (m_hTrackedCamera == INVALID_TRACKED_CAMERA_HANDLE)
    {
        std::cout<<"AcquireVideoStreamingService() Failed!\n";
        return 1;
    }

    while (true)
    {
        //std::cout << imgBuffer.size() << "\n";
        if (imgBuffer.size() == 100)
            break;

        if (!m_pVRTrackedCamera || !m_hTrackedCamera)
        {
            std::cout << "No Camera Found\n";
            return 1;
        }

        // get the frame header only
        vr::CameraVideoStreamFrameHeader_t frameHeader;
        vr::EVRTrackedCameraError nCameraError = m_pVRTrackedCamera->GetVideoStreamFrameBuffer(m_hTrackedCamera, vr::VRTrackedCameraFrameType_Undistorted, nullptr, 0, &frameHeader, sizeof(frameHeader));
        if (nCameraError != vr::VRTrackedCameraError_None)
        {
            std::cout << "No Tracked Camera Found\n";
            return 1;
        }

        if (frameHeader.nFrameSequence == m_nLastFrameSequence)
        {
            continue;
        }

        if (m_nCameraFrameBufferSize / (m_nCameraFrameHeight*m_nCameraFrameWidth) == 4)
        {
            image.create(m_nCameraFrameHeight, m_nCameraFrameWidth, CV_8UC4);
        }
        else if (m_nCameraFrameBufferSize / (m_nCameraFrameHeight*m_nCameraFrameWidth) == 3)
        {
            image.create(m_nCameraFrameHeight, m_nCameraFrameWidth, CV_8UC3);
        }
        else
        {
            std::cout << "Error in Frame Size\n";
        }
        
        // Frame has changed, do the more expensive frame buffer copy
        nCameraError = m_pVRTrackedCamera->GetVideoStreamFrameBuffer(m_hTrackedCamera, vr::VRTrackedCameraFrameType_Undistorted, (uint8_t*)image.data, m_nCameraFrameBufferSize, &frameHeader, sizeof(frameHeader));
        if (nCameraError != vr::VRTrackedCameraError_None)
        {
            std::cout << "No Tracked Camera Found\n";
            return 1;
        }

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        imgBuffer.push_back(image.clone());

        m_nLastFrameSequence = frameHeader.nFrameSequence;

        if(!frameHeader.standingTrackedDevicePose.bPoseIsValid || frameHeader.standingTrackedDevicePose.eTrackingResult != vr::TrackingResult_Running_OK)
            cv::putText(image, "Invalid Pose: ", cv::Point(200, 20), cv::FONT_HERSHEY_PLAIN, 1, 255);
        else
        {
            cv::putText(image, "Valid Pose: ", cv::Point(200, 20), cv::FONT_HERSHEY_PLAIN, 1, 255);

            Mat pose(3, 4, CV_32F);
            for (int i = 0; i < 3; i++)
            {
                // emit the matrix
                vr::HmdMatrix34_t *pMatrix = &frameHeader.standingTrackedDevicePose.mDeviceToAbsoluteTracking;
                std::stringstream str1;
                str1.precision(3);
                str1 << pMatrix->m[i][0] << " " << pMatrix->m[i][1] << " " << pMatrix->m[i][2] << " " << pMatrix->m[i][3];
                cv::putText(image, str1.str(), cv::Point(310, 20 * (i + 1)), cv::FONT_HERSHEY_PLAIN, 1, 255);
                pose.at<float>(i, 0) = pMatrix->m[i][0];
                pose.at<float>(i, 1) = pMatrix->m[i][1];
                pose.at<float>(i, 2) = pMatrix->m[i][2];
                pose.at<float>(i, 3) = pMatrix->m[i][3];
            }
            poseBuffer.push_back(pose.clone());

            cv::imshow("Image", image);
            cv::waitKey(1);
            //cv::imshow("Mask", validMask);
        }
    }

    Mat cameraMatrix(3, 3, CV_32F);
    setIdentity(cameraMatrix);
    cameraMatrix.at<float>(0, 0) = m_nFocalLength.v[0];
    cameraMatrix.at<float>(1, 1) = m_nFocalLength.v[1];

    cameraMatrix.at<float>(0, 2) = m_nCenter.v[0];
    cameraMatrix.at<float>(1, 2) = m_nCenter.v[1];

    FileStorage fs("F:/Data/Vive/2/Poses.xml", FileStorage::WRITE);
    fs << "nr_of_frames" << (int)(imgBuffer.size());
    fs << "Camera_Matrix" << cameraMatrix;

    for (int i = 0; i < imgBuffer.size(); ++i)
    {
        std::stringstream str1, str2;
        str1 << "F:/Data/Vive/2/" << i << ".png";
        cv::imwrite(str1.str(), imgBuffer[i]);

        str2 << "Pose_Matrix_" << i;
        fs << str2.str() << poseBuffer[i];
    }
    

    return 0;
}
