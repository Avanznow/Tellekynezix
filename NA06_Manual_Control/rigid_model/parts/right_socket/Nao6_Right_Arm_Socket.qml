import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: altavoz_035_material
        objectName: "ALTAVOZ.035"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Right_Arm_Socket_obj
        objectName: "Nao6_Right_Arm_Socket.obj"
        Model {
            id: nao6_Right_Arm_Socket
            objectName: "Nao6_Right_Arm_Socket"
            source: "meshes/nao6_Right_Arm_Socket_mesh.mesh"
            materials: [
                altavoz_035_material
            ]
        }
    }

    // Animations:
}
