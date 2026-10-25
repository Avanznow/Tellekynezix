import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: altavoz_011_material
        objectName: "ALTAVOZ.011"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Arm_Socket_obj
        objectName: "Nao6_Left_Arm_Socket.obj"
        Model {
            id: nao6_Left_Arm_Socket
            objectName: "Nao6_Left_Arm_Socket"
            source: "meshes/nao6_Left_Arm_Socket_mesh.mesh"
            materials: [
                altavoz_011_material
            ]
        }
    }

    // Animations:
}
