import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_TRANSLUCIDO_006_material
        objectName: "PLASTICO_TRANSLUCIDO.006"
        baseColor: "#ffccc49f"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Hand_Bottom_Finger_obj
        objectName: "Nao6_Left_Hand_Bottom_Finger.obj"
        Model {
            id: nao6_Left_Hand_Bottom_Finger
            objectName: "Nao6_Left_Hand_Bottom_Finger"
            source: "meshes/nao6_Left_Hand_Bottom_Finger_mesh.mesh"
            materials: [
                plastico_TRANSLUCIDO_006_material
            ]
        }
    }

    // Animations:
}
