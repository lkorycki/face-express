#include "Logger.h"

Logger::Logger()
{
}

void Logger::show(const double* fv)
{
    cls();
    showHeader();
    showFeatureVector(fv);
}

void Logger::showHeader()
{
    cls();
    cout << "Running: face-express-prod v0.0.0\n";
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "Image input: default camera\n";
    cout << "...\n";
}

void Logger::showFeatureVector(const double* fv)
{
    cout << "---------- Feature vector ----------\n";

    cout << "LEFT EYE WIDTH: \t\t" << fv[0] << endl;
    cout << "LEFT EYE HEIGHT: \t\t" << fv[1] << endl;
    cout << "RIGHT EYE WIDTH: \t\t" << fv[2] << endl;
    cout << "RIGHT EYE HEIGHT: \t\t" << fv[3] << endl;

    cout << "(LEFT CENTER) EYE - EYEBROW: \t" << fv[4] << endl;
    cout << "(LEFT OUTER) EYE - EYEBROW: \t" << fv[5] << endl;
    cout << "(LEFT INNER) EYE - EYEBROW: \t" << fv[6] << endl;
    cout << "(RIGHT CENTER) EYE - EYEBROW: \t" << fv[7] << endl;
    cout << "(RIGHT OUTER) EYE - EYEBROW: \t" << fv[8] << endl;
    cout << "(RIGHT INNER) EYE - EYEBROW: \t" << fv[9] << endl;

    cout << "MOUTH WIDTH: \t\t\t" << fv[10] << endl;
    cout << "MOUTH HEIGHT: \t\t\t" << fv[11] << endl;
    cout << "LOWER LIP - NOSE TIP: \t\t" << fv[12] << endl;
    cout << "LEFT MOUTH CORNER - NOSE TIP: \t" << fv[13] << endl;
    cout << "RIGHT MOUTH - NOSE TIP: \t" << fv[14] << endl;

    cout << "TEETH W2B_PARAM: \t\t" << fv[15] << endl;

    cout << "------------------------------------\n";
}

void Logger::cls()
{
    #ifdef WINDOWS
        std::system("cls");
    #else
        std::system ("clear");
    #endif
}

void Logger::writeToFile(const double* fv, string path)
{
    ofstream out;
    out.open(path);

    for(int i = 0; i < 16; i++)
    {
       stringstream ss; ss << i;
       out << "X" << ss.str() << "\t" << fv[i] << endl;
    }

    out.close();
}

string Logger::getTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);

    string dst(buf);
    boost::replace_all(dst, ":", "");
    return dst;
}

Logger::~Logger()
{
}
