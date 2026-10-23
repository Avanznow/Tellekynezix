import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_material
        objectName: "PLASTICO_BLANCO"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: metalico_naranja_material
        objectName: "Metalico_naranja"
        baseColor: "#ffcc1104"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: altavoz_001_material
        objectName: "ALTAVOZ.001"
        baseColor: "#ff727272"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: ojos_material
        objectName: "OJOS"
        baseColor: "#ff8fccbb"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: ojos_NEGRO_material
        objectName: "OJOS_NEGRO"
        baseColor: "#ff020202"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: gris_material
        objectName: "GRIS"
        baseColor: "#ff151515"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Head_obj
        objectName: "Nao6_Head.obj"
        Model {
            id: nao6_Head
            objectName: "Nao6_Head"
            source: "meshes/nao6_Head_mesh.mesh"
            materials: [
                plastico_BLANCO_material,
                metalico_naranja_material,
                altavoz_001_material,
                ojos_material,
                ojos_NEGRO_material,
                gris_material
            ]
        }
    }

    // Animations:
}
