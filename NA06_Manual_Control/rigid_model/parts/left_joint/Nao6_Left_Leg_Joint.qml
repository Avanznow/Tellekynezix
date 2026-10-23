import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: altavoz_023_material
        objectName: "ALTAVOZ.023"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Leg_Joint_obj
        objectName: "Nao6_Left_Leg_Joint.obj"
        Model {
            id: nao6_Left_Leg_Joint
            objectName: "Nao6_Left_Leg_Joint"
            source: "meshes/nao6_Left_Leg_Joint_mesh.mesh"
            materials: [
                altavoz_023_material
            ]
        }
    }

    // Animations:
}
