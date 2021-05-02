#include "npy_parser.hpp"

double
decode_integer(const char *string)
{
    // Little-endian two's complement
    uint64_t *le_number = (uint64_t*)  string;
    uint64_t  h_number  =              le64toh(*le_number);
    int64_t  *interm    = (int64_t*)  &h_number;

    return (double) *interm;
}

double
decode_double(const char *string)
{
    // Little-endian IEEE 754 64-bit double
    uint64_t *le_number = (uint64_t*)  string;
    uint64_t  h_number  =              le64toh(*le_number);
    double   *number    = (double*)   &h_number;

    return *number;
}

bool
check_header(std::ifstream *fobj)
{
    char header[6];
    header[5] = '\0';
    fobj->seekg(1);
    fobj->read(header, 5);
    return (std::strcmp(header, "NUMPY") == 0);
}

void
populate_metadata(std::ifstream   *fobj,
                  Metadata *metadata)
{
    char dict_string[119];
    dict_string[118] = '\0';
    fobj->seekg(10);
    fobj->read(dict_string, 118);

    char expect = 0; // 0: start, 1: key or end, 2: value, 3: nothing
    /* special characters:
     * {    begins dictionary, announces key
     * }    ends dictionary
     * :    announces value
     * ,    announces key or dict. end
     * '    ignore
     * ' '  ignore
     */
    bool popen = false; // value parenthesis open
    char buf[100]; buf[0] = '\0';
    char buf_pos = 0;
    char last_key[100] = {};
    char last_val[100] = {};
    for (uint16_t i = 0; i < 118; i++) {
        switch(expect) {
        case 0: // expecting opening bracket
            if (dict_string[i] == '{') {
                expect = 1;
                buf[0] = '\0';
                buf_pos = 0;
            }
            break;
        case 1: // expecting key, assign operator or closing bracket
            if (dict_string[i] == '}') {
                expect = 3;
            } else if (dict_string[i] == ':') {
                buf[buf_pos] = '\0';
                std::strcpy(last_key, buf);

                expect = 2;
                buf[0] = '\0';
                buf_pos = 0;
            } else if ((dict_string[i] != ' ') && (dict_string[i] != '\'')) {
                buf[buf_pos] = dict_string[i];
                buf_pos++;
            }
            break;
        case 2: // expecting value or comma operator
            if ((dict_string[i] == ',') && (!popen)) {
                buf[buf_pos] = '\0';
                std::strcpy(last_val, buf);
                buf[0] = '\0';
                buf_pos = 0;

                // Process values
                // Read number encoding
                if        (std::strcmp(last_key, "descr") == 0) {
                    if        (std::strcmp(last_val, "<i8") == 0) {
                        metadata->num_type = 1;
                    } else if (std::strcmp(last_val, "<f8") == 0) {
                        metadata->num_type = 2;
                    }
                // Get rows and cols
                } else if (std::strcmp(last_key, "shape") == 0) {
                    for (uint8_t j = 0; j < strlen(last_val); j++) {
                        if        (last_val[j] == ' ') {
                            continue;
                        } else if (last_val[j] == '(') {
                            continue;
                        } else if (last_val[j] == ',') {
                            buf[buf_pos] = '\0';
                            if (strlen(buf) == 0)
                                metadata->rows = 1;
                            else
                                metadata->rows = atoi(buf);
                            buf[0] = '\0';
                            buf_pos = 0;
                        } else if (last_val[j] == ')') {
                            buf[buf_pos] = '\0';
                            if (strlen(buf) == 0)
                                metadata->cols = 1;
                            else
                                metadata->cols = atoi(buf);
                            buf[0] = '\0';
                            buf_pos = 0;
                        } else {
                            buf[buf_pos] = last_val[j];
                            buf_pos++;
                        }
                    }
                }

                expect = 1;
                buf[0] = '\0';
                buf_pos = 0;
            } else if ((dict_string[i] != ' ') && (dict_string[i] != '\'')) {
                if (dict_string[i] == '(')
                    popen = true;
                else if (dict_string[i] == ')')
                    popen = false;
                buf[buf_pos] = dict_string[i];
                buf_pos++;
            }
            break;
        }
    }

}

template <class v_type>
void
populate_vector(std::ifstream        *npy_fobj,
                Metadata             *metadata,
                std::vector<v_type>  *targetvec)
{
    if (!check_header(npy_fobj))
        throw NPYPARSER_HEADER_ERROR;

    populate_metadata(npy_fobj, metadata);
    uint64_t elem_count = metadata->rows*metadata->cols;

    char *numbuf = new char[8*elem_count+1];
    numbuf[8*elem_count+1] = '\0';
    npy_fobj->seekg(128);
    npy_fobj->read(numbuf,8*elem_count);
    
    *targetvec = std::vector<v_type>(elem_count, (v_type) 0.0);

    double (*decoder_func)(const char *string);
    if      (metadata->num_type == 1)
        decoder_func = &decode_integer;
    else if (metadata->num_type == 2)
        decoder_func = &decode_double;

    int16_t irow = 0;
    int16_t icol = 0;
    for (uint64_t i = 0; i < elem_count; i++) {
        (*targetvec)[i] = (v_type) decoder_func((char*) &numbuf[8*i]);
    }
    delete numbuf;
}
template void populate_vector<uint64_t> (std::ifstream*, 
                                         Metadata*, 
                                         std::vector<uint64_t>*);
template void populate_vector<int64_t>  (std::ifstream*, 
                                         Metadata*, 
                                         std::vector<int64_t>*);
template void populate_vector<float>    (std::ifstream*, 
                                         Metadata*, 
                                         std::vector<float>*);
template void populate_vector<double>   (std::ifstream*, 
                                         Metadata*, 
                                         std::vector<double>*);


void populate_lookup_table(std::ifstream *lut_fobj,
                           Eigen::MatrixXd *lut)
{
    if (!check_header(lut_fobj))
        throw NPYPARSER_HEADER_ERROR;

    Metadata metadata;
    populate_metadata(lut_fobj, &metadata);
    uint64_t elem_count = metadata.rows*metadata.cols;

    // Read numbers, populate matrix
    // -----------------------------
    char *numbuf = new char[8*elem_count+1];
    numbuf[8*elem_count+1] = '\0';
    lut_fobj->seekg(128);
    lut_fobj->read(numbuf,8*elem_count);

    *lut = Eigen::MatrixXd::Zero(metadata.rows, metadata.cols);

    double (*decoder_func)(const char *string);
    if      (metadata.num_type == 1)
        decoder_func = &decode_integer;
    else if (metadata.num_type == 2)
        decoder_func = &decode_double;

    int16_t irow = 0;
    int16_t icol = 0;
    for (uint64_t i = 0; i < elem_count; i++) {
        icol = i % metadata.cols;
        irow = i / metadata.cols;
        (*lut)(irow, icol) = decoder_func((char*) &numbuf[8*i]);
    }
    delete numbuf;
}

void
populate_lookup_row(std::ifstream *lut_fobj,
                    Eigen::VectorXd *row,
                    const uint16_t irow)
{
    if (!check_header(lut_fobj))
        throw NPYPARSER_HEADER_ERROR;

    Metadata metadata;
    populate_metadata(lut_fobj, &metadata);

    char *numbuf = new char[8*metadata.cols+1];
    numbuf[8*metadata.cols+1] = '\0';
    lut_fobj->seekg(128+8*irow*metadata.cols);
    lut_fobj->read(numbuf,8*metadata.cols);

    if (irow >= metadata.rows)
        throw NPYPARSER_INDEX_ERROR;

    *row = Eigen::VectorXd::Zero(metadata.cols);

    double (*decoder_func)(const char *string);
    if      (metadata.num_type == 1)
        decoder_func = &decode_integer;
    else if (metadata.num_type == 2)
        decoder_func = &decode_double;

    for (uint64_t i = 0; i < metadata.cols; i++) {
        (*row)[i] = decoder_func((char*) &numbuf[8*i]);
    }
    delete numbuf;
}
