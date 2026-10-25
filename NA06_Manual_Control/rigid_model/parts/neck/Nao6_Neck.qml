import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: altavoz_025_material
        objectName: "ALTAVOZ.025"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Neck_obj
        objectName: "Nao6_Neck.obj"
        Model {
            id: nao6_Neck
            objectName: "Nao6_Neck"
            source: "meshes/nao6_Neck_mesh.mesh"
            materials: [
                altavoz_025_material
            ]
        }
    }

    // Animations:
}
