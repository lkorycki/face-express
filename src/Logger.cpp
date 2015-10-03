#include "Logger.h"

Logger::Logger()
{
}

void Logger::show(const double* fv, bool toFile)
{
    cls();
    showHeader();
    showFeatureVector(fv, toFile);
}

void Logger::showHeader()
{
    cls();
    cout << "Running: face-express-prod v0.0.0\n";
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "Image input: default camera\n";
    cout << "...\n";
}

void Logger::showFeatureVector(const double* fv, bool toFile)
{
    cout << "---------- Feature vector ----------\n";

    cout << "LEFT EYE WIDTH: " << fv[0] << endl;
    cout << "LEFT EYE HEIGHT: " << fv[1] << endl;
    cout << "RIGHT EYE WIDTH: " << fv[2] << endl;
    cout << "RIGHT EYE HEIGHT: " << fv[3] << endl;

    cout << "(LEFT CENTER) EYE - EYEBROW: " << fv[4] << endl;
    cout << "(LEFT OUTER) EYE - EYEBROW: " << fv[5] << endl;
    cout << "(LEFT INNER) EYE - EYEBROW: " << fv[6] << endl;
    cout << "(RIGHT CENTER) EYE - EYEBROW: " << fv[7] << endl;
    cout << "(RIGHT OUTER) EYE - EYEBROW: " << fv[8] << endl;
    cout << "(RIGHT INNER) EYE - EYEBROW: " << fv[9] << endl;

    cout << "MOUTH WIDTH: " << fv[10] << endl;
    cout << "MOUTH HEIGHT: " << fv[11] << endl;
    cout << "LOWER LIP - NOSE TIP: " << fv[12] << endl;
    cout << "LEFT MOUTH CORNER - NOSE TIP: " << fv[13] << endl;
    cout << "RIGHT MOUTH - NOSE TIP: " << fv[14] << endl;

    cout << "TEETH W2B_PARAM: " << fv[15] << endl;

    cout << "------------------------------------\n";

    // TODO: wrtiting to file
}

void Logger::cls()
{
    #ifdef WINDOWS
        std::system("cls");
    #else
        std::system ("clear");
    #endif
}

Logger::~Logger()
{
}
