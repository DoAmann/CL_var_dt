#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <sstream>
#include <iomanip>

#include <stdio.h>
#include <math.h>

#include "Parameters.h"


using namespace std;

void modifyLineInFile(const string& filename, int lineNumber, const string& newContent) {
    ifstream fileIn(filename);
    if (!fileIn) {
        cerr << "Errore nell'apertura del file: " << filename << endl;
        return;
    }

    vector<string> lines;
    string line;
    while (getline(fileIn, line)) {
        lines.push_back(line);
    }
    fileIn.close();

    if (lineNumber < 1 || lineNumber > lines.size()) {
        cerr << "Numero di riga non valido. Il file ha " << lines.size() << " righe." << endl;
        return;
    }

    // Modifica la riga specificata
    lines[lineNumber - 1] = newContent;

    // Scrive le modifiche nel file
    ofstream fileOut(filename);
    if (!fileOut) {
        cerr << "Errore nell'apertura del file per la scrittura: " << filename << endl;
        return;
    }

    for (const auto& l : lines) {
        fileOut << l << endl;
    }
    fileOut.close();
}

int main(int argc, char *argv[]) {
    
    ostringstream oss;
    
    
    double term_exp = D / pow(2, 0.5);
    double prefactor = M / (3 * D * pow(2 * M_PI, 0.5));
    double term1 = 3 * pow(M_PI / 2, 0.5) * D;
    double sin2 = (sin(SIGMA)*sin(SIGMA));
    double cos2 = (cos(SIGMA)*cos(SIGMA));

    double g_2D = (G / (D * pow(M_PI * 2 ,  0.5)));
    
    string filename;
    filename = argv[1];
    //cout << filename;
    int lineNumber = 45;
    string newContent;

    oss << scientific << setprecision(60) << term_exp; 
    newContent = "#define TERM_EXP "+ oss.str(); ;
    modifyLineInFile(filename, lineNumber, newContent);
    oss.str(""); 
    oss.clear();


    oss << scientific << setprecision(60) << prefactor; 
    newContent = "#define PREFACTOR "+ oss.str(); ;
    modifyLineInFile(filename, lineNumber+1, newContent);
    oss.str(""); 
    oss.clear();

    oss << scientific << setprecision(60) << term1; 
    newContent = "#define TERM1 "+ oss.str(); ;
    modifyLineInFile(filename, lineNumber+2, newContent);
    oss.str(""); 
    oss.clear();

    oss << scientific << setprecision(60) << sin2; 
    newContent = "#define SIN2 "+ oss.str(); ;
    modifyLineInFile(filename, lineNumber+3, newContent);
    oss.str(""); 
    oss.clear();

    oss << scientific << setprecision(60) << cos2; 
    newContent = "#define COS2 "+ oss.str(); ;
    modifyLineInFile(filename, lineNumber+4, newContent);
    oss.str(""); 
    oss.clear();

    oss << scientific << setprecision(60) << g_2D; 
    newContent = "#define G_2D "+ oss.str(); ;
    modifyLineInFile(filename, lineNumber+5, newContent);
    oss.str(""); 
    oss.clear();
    
    cout << "Parameters updated." << endl;

    return 0;
}


/*
double exponential = exp( (rho_2D*rho_2D * D*D / 2.) + log(erfc(rho_2D * term_exp)) );


factor=   prefactor  * 
    (   (-1 + term1 * rho_2D * ((px*px) / (p2 - pz*pz)) * exponential  ) * (sin2) + 
        (2 - term1 * rho_2D * exponential ) * (cos2) 	) ;
*/


