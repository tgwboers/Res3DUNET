#include <Python.h>
#include "numpy/arrayobject.h"
#include "geodesic_distance_2d.h"
#include "geodesic_distance_3d.h"
#include <iostream>
using namespace std;

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

// example to use numpy object: http://blog.debao.me/2013/04/my-first-c-extension-to-numpy/
// write a c extension ot Numpy: http://folk.uio.no/hpl/scripting/doc/python/NumPy/Numeric/numpy-13.html
static PyObject *
geodesic2d_fast_marching_wrapper(PyObject *self, PyObject *args)
{
    PyObject *I=NULL, *Seed=NULL;
    PyArrayObject *arr_I=NULL, *arr_Seed=NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &I, &Seed)) return NULL;
    
    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I, NPY_FLOAT32, NPY_IN_ARRAY);
    if (arr_I == NULL) return NULL;
    
    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(Seed, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_Seed == NULL) return NULL;
    
    
    int nd = PyArray_NDIM(arr_I);   //number of dimensions
    npy_intp * shape = PyArray_DIMS(arr_I);  // npy_intp array of length nd showing length in each dim.
    npy_intp * shape_seed = PyArray_DIMS(arr_Seed);
    cout<<"input shape: ";
    for(int i=0; i<nd; i++)
    {
        cout<<shape[i]<<" ";
        if(shape[i]!=shape_seed[i])
        {
            cout<<"input shape does not match"<<endl;
            return NULL;
        }
    }
    cout<<std::endl;
    
    int output_shape[2];
    output_shape[0] = shape[0];
    output_shape[1] = shape[1];
    PyArrayObject * distance = (PyArrayObject*)  PyArray_FromDims(2, output_shape, NPY_FLOAT32);
    geodesic2d_fast_marching((const float *)arr_I->data, (const unsigned char *)arr_Seed->data, (float *) distance->data, shape[0], shape[1]);
    
    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    Py_INCREF(distance);
    return PyArray_Return(distance);
}

static PyObject *
geodesic2d_raster_scan_wrapper(PyObject *self, PyObject *args)
{
    PyObject *I=NULL, *Seed=NULL;
    PyArrayObject *arr_I=NULL, *arr_Seed=NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &I, &Seed)) return NULL;
    
    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I, NPY_FLOAT32, NPY_IN_ARRAY);
    if (arr_I == NULL) return NULL;
    
    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(Seed, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_Seed == NULL) return NULL;
    
    
    int nd = PyArray_NDIM(arr_I);   //number of dimensions
    npy_intp * shape = PyArray_DIMS(arr_I);  // npy_intp array of length nd showing length in each dim.
    npy_intp * shape_seed = PyArray_DIMS(arr_Seed);
    cout<<"input shape: ";
    for(int i=0; i<nd; i++)
    {
        cout<<shape[i]<<" ";
        if(shape[i]!=shape_seed[i])
        {
            cout<<"input shape does not match"<<endl;
            return NULL;
        }
    }
    cout<<std::endl;
    
    int output_shape[2];
    output_shape[0] = shape[0];
    output_shape[1] = shape[1];
    PyArrayObject * distance = (PyArrayObject*)  PyArray_FromDims(2, output_shape, NPY_FLOAT32);
    geodesic2d_raster_scan((const float *)arr_I->data, (const unsigned char *)arr_Seed->data, (float *) distance->data, shape[0], shape[1]);
    
    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    Py_INCREF(distance);
    return PyArray_Return(distance);
}

static PyObject *
geodesic3d_fast_marching_wrapper(PyObject *self, PyObject *args)
{
    PyObject *I=NULL, *Seed=NULL;
    PyArrayObject *arr_I=NULL, *arr_Seed=NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &I, &Seed)) return NULL;
    
    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I, NPY_FLOAT32, NPY_IN_ARRAY);
    if (arr_I == NULL) return NULL;
    
    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(Seed, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_Seed == NULL) return NULL;
    
    
    int nd = PyArray_NDIM(arr_I);   //number of dimensions
    npy_intp * shape = PyArray_DIMS(arr_I);  // npy_intp array of length nd showing length in each dim.
    npy_intp * shape_seed = PyArray_DIMS(arr_Seed);
    cout<<"input shape: ";
    for(int i=0; i<nd; i++)
    {
        cout<<shape[i]<<" ";
        if(shape[i]!=shape_seed[i])
        {
            cout<<"input shape does not match"<<endl;
            return NULL;
        }
    }
    cout<<std::endl;
    
    int output_shape[3];
    output_shape[0] = shape[0];
    output_shape[1] = shape[1];
    output_shape[2] = shape[2];
    PyArrayObject * distance = (PyArrayObject*)  PyArray_FromDims(3, output_shape, NPY_FLOAT32);
    geodesic3d_fast_marching((const float *)arr_I->data, (const unsigned char *)arr_Seed->data, (float *) distance->data,
                        shape[0], shape[1], shape[2]);
    
    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    Py_INCREF(distance);
    return PyArray_Return(distance);
}

static PyObject *
geodesic3d_raster_scan_wrapper(PyObject *self, PyObject *args)
{
    PyObject *I=NULL, *Seed=NULL;
    float lambda, iteration;
    PyArrayObject *arr_I=NULL, *arr_Seed=NULL;
    
    if (!PyArg_ParseTuple(args, "OOff", &I, &Seed, &lambda, &iteration)) return NULL;
    
    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I, NPY_FLOAT32, NPY_IN_ARRAY);
    if (arr_I == NULL) return NULL;
    
    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(Seed, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_Seed == NULL) return NULL;
    
    
    int nd = PyArray_NDIM(arr_I);   //number of dimensions
    npy_intp * shape = PyArray_DIMS(arr_I);  // npy_intp array of length nd showing length in each dim.
    npy_intp * shape_seed = PyArray_DIMS(arr_Seed);
    for(int i=0; i<nd; i++)
    {
        if(shape[i]!=shape_seed[i])
        {
            cout<<"input shape does not match"<<endl;
            return NULL;
        }
    }
    
    int output_shape[3];
    output_shape[0] = shape[0];
    output_shape[1] = shape[1];
    output_shape[2] = shape[2];
    PyArrayObject * distance = (PyArrayObject*)  PyArray_FromDims(3, output_shape, NPY_FLOAT32);
    geodesic3d_raster_scan((const float *)arr_I->data, (const unsigned char *)arr_Seed->data, (float *) distance->data,
                        shape[0], shape[1], shape[2], lambda, (int) iteration);
    
    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    Py_INCREF(distance);
    return PyArray_Return(distance);
}

static PyMethodDef Methods[] = {
    {"geodesic2d_fast_marching",  geodesic2d_fast_marching_wrapper, METH_VARARGS, "computing 2d geodesic distance"},
    {"geodesic2d_raster_scan",  geodesic2d_raster_scan_wrapper, METH_VARARGS, "computing 2d geodesic distance"},
    {"geodesic3d_fast_marching",  geodesic3d_fast_marching_wrapper, METH_VARARGS, "computing 3d geodesic distance"},
    {"geodesic3d_raster_scan",  geodesic3d_raster_scan_wrapper, METH_VARARGS, "computing 3d geodesic distance"},
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "geodesic_distance",
        NULL,
        sizeof(struct module_state),
        Methods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};
#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_geodesic_distance(void)

#else
#define INITERROR return

void
initgeodesic_distance(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("geodesic_distance", Methods);
#endif
    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("geodesic_distance.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    import_array();
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
